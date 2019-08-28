import os
import sys
import h5py
import numpy as np
import pandas as pd

import torch
import torch.utils.data

class MPIIFaceGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        """
            Load hdf5 file header into memory.
        """
        # file names like "p00_0.h5"
        self.path = os.path.join(dataset_dir, '{}.h5'.format(subject_id))
        self.dataset = None
        with h5py.File(self.path, 'r') as file:
            self.length = len(file["data"])

    def __getitem__(self, index):
        """
            Load indexed image to memory, then transformations in its shape.
            Gaze -> #TODO see what returns
        """
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.dataset == None:
            self.dataset = h5py.File(self.path, 'r')
            self.images = self.dataset['data']
            self.gazes = self.dataset['label']

        # original shape: [channel[BGR], x, y]
        # [2, 1, 0] reshapes BGR to RGB [channel[RGB], x, y]
        # couple of transposes to put image to correct representation
        # first one puts channel dim last and brings forward the y dim [y, x, channel[RGB]]
        # second brings forward the x dim and puts y dim second [x, y, channel[RGB]]
        #img = torch.from_numpy((self.images[index][[2, 1, 0],:,:])).transpose(2, 0).transpose(1, 0)
        img = torch.from_numpy((self.images[index][[2, 1, 0],:,:])) # expected input [32, 3, 448, 448]
        #img = torch.unsqueeze(torch.from_numpy((self.images[index][[2, 1, 0],:,:])).transpose(2, 0).transpose(1, 0), 1)
        gaze = torch.from_numpy(self.gazes[index][0:2])
        return img, gaze

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu):
    """
        Mounts the train and test loaders to all .h5 files from the dataset (15 subjects, 2 files each).
        Names like: 'p00_0.h5', 'p00_1.h5'
    """
    assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)

    base_subject_ids = ['p{:02}'.format(i) for i in range(15)]
    base_test_subject_id = base_subject_ids[test_subject_id]

    test_subject_ids = []
    subject_ids = []
    for i, subject_id in enumerate(base_subject_ids):
        for file_part in range(2):
            subject_ids.append(subject_id + '_' + str(file_part))

    for file_part in range(2):
        test_subject_ids.append(base_test_subject_id + '_' + str(file_part))

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIFaceGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
        if subject_id not in test_subject_ids
    ])

    test_dataset = torch.utils.data.ConcatDataset([
        MPIIFaceGazeDataset(i, dataset_dir) for i in test_subject_ids
    ])

    assert len(train_dataset) == 42000
    assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
