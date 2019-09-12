# /usr/local/python3
# -*- coding: utf-8 -*-

# https://github.com/ti-ginkgo/MPIIFaceGaze/blob/master/main.py
import os
import sys
import argparse
import ast
import json
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils

from models import GazeNet
from dataloader import get_loader
from utils import AverageMeter, compute_angle_error

global_step = 0

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)


def validate(args, epoch, model, criterion, test_loader):
    global global_step

    logger.info('Test {}'.format(epoch))
    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    for step, (images, gazes) in enumerate(test_loader):
        if args.use_cuda:
            images = images.cuda()
            gazes = gazes.cuda()

        with torch.no_grad():
            outputs = model(images)
        loss = criterion(outputs, gazes)

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)

        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        logger.info('Epoch {} Loss {:.4f} AngleError {:.2f}'.format(
            epoch, loss_meter.avg, angle_error_meter.avg
        ))

        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))

        return angle_error_meter.avg


def train(args, epoch, model, optimizer, criterion, train_loader):
    global global_step

    logger.info('Train {}'.format(epoch))
    model.train()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    for step, (images, gazes) in enumerate(train_loader):
        global_step += 1

        if args.use_cuda:
            images = images.cuda()
            gazes = gazes.cuda()

        # zero optimizer's gradient
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, gazes)
        loss.backward()

        optimizer.step()
        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        print("Epoch {} Step {}/{} Loss {:.4f} Angle Error {:.2f}".format(epoch, step, len(train_loader), loss_meter.avg, angle_error_meter.avg))

        if step % 10 == 0:
            logger.info('Epoch {} Step {}/{}\t'
                        'Loss {:.4f} ({:.4f})\t'
                        'Angle Error {:.2f} ({:.2f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            angle_error_meter.val,
                            angle_error_meter.avg,
                        ))
            print("Gaze[0:2]: ", gazes[0:2])
            print("Output[0:2]: ", outputs[0:2])
        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))

def main(args):
    """
     GazeNet was implemented using the Caffe library (Jia et al., 2014).
    We used the weights of the 16-layer VGGNet (Simonyan and Zisserman, 2015) pretrained on ImageNet for all our evaluations, and fine-tuned the whole network in
    15,000 iterations with a batch size of 256 on the training set. We used the Adam
    solver (Kingma and Ba, 2015) with the two momentum values set to β1 = 0.9 and
    β2 = 0.95. An initial learning rate of 0.00001 was used and multiplied by 0.1 after
    every 5,000 iterations.
    """
    global device

    if args.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(json.dumps(vars(args), indent=2))

    # set random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # model configs json
    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = get_loader(
        args.dataset, args.test_id, args.batch_size, args.num_workers, True
    )

    model = GazeNet()
    if args.use_cuda:
        model.cuda()

    #criterion = nn.MSELoss(size_average=True)
    criterion = nn.L1Loss()
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
    """

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=ast.literal_eval(args.milestones), gamma=args.lr_decay
    )

    validate(args, 0, model, criterion, val_loader)

    for epoch in range(1, args.epochs+1):

        # train and get validation error
        train(args, epoch, model, optimizer, criterion, train_loader)
        angle_error = validate(args, epoch, model, criterion, val_loader)

        # lr decay
        lr_scheduler.step()

        state = OrderedDict([
            ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('angle_error', angle_error),
        ])

        model_path = os.path.join(outdir, 'model_state.sth')
        torch.save(state, model_path)

def arg_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/nvidia/HDLuiza/Dataset/Gaze/MPIIFaceGaze_normalizad')
    parser.add_argument('--test_id', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--milestones', type=str, default='[5, 10, 20, 30]')
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--use_cuda', type=bool, default=True)

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(arg_parser(sys.argv[1:]))
