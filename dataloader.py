# -*- coding: utf-8 -*-
# source: https://github.com/fab-jul/hdf5_dataloader

import os
import sys
import h5py
import glob
import numbers
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

default_opener = lambda p_: h5py.File(p_, 'r')

class HDF5Dataset(Dataset):
    def __init__(self, file_ps,
                 custom_getitem,
                 files_and_shards,
                 transform=None,
                 transform_label=None,
                 shuffle_shards=True,
                 opener=default_opener,
                 seed=42):
        """
        Parameters:
            - file_ps: list of file paths to .hdf5 files. Last (alphabetically)
            file is expected to contain lessimages.
            - custom_getitem: custom function defined by user to get data from
            the hdf5 file. It's inputs are be opened hdf5 file and sampled index.
            - files_and_shards: dictionary containing entries such as
                    {file1: num_shards1, ..., fileN: num_shardsN}
            - transform: transformation to apply to read HDF5 sample.
            - transform_label: tranformation to apply to read HDF5 label.
            - shuffle_shards: if true, shards are shuffled with seed
        """

        if len(file_ps) == 0 or not all(os.path.isfile(p) for p in file_ps):
            raise ValueError('Expected list of paths to HDF5 files, got {}'.format(file_ps))
        self.opener = opener
        self.ps, self.num_per_shard = HDF5Dataset.filter_smaller_shards(file_ps)
        if shuffle_shards:
            r = random.Random(seed)
            r.shuffle(self.ps)
        self.transform = transform
        self.transform_label = transform_label

        ## Custom function defined by user to handle
        # return of data from hdf5 file.
        self.custom_getitem = custom_getitem

    def __len__(self):
        return len(self.ps) * self.num_per_shard

    def __getitem__(self, index):
        shard_idx = index // self.num_per_shard
        idx_in_shard = index % self.num_per_shard
        shard_p = self.ps[shard_idx]
        with self.opener(shard_p) as f:
            item = self.custom_getitem(f, idx_in_shard)
        if self.transform is not None:
            item['sample'] = self.transform(img)
        if self.transform_label is not None:
            item['label'] = self.transform(label)
        return item

    @staticmethod
    def filter_smaller_shards(file_ps, opener=default_opener):
        """
        Filter away the (alphabetically) last shard, which is assumed to be smaller. This function also double checks
        that all other shards have the same number of entries.
        Parameters:
            - file_ps: list of .hdf5 files, does not have to be sorted.
            - opener: function to open shards
        Return:
            - tuple (ps, num_per_shard), where
                -> ps = filtered file paths,
                -> num_per_shard = number of entries in all of the shards in `ps`
        """
        assert file_ps, 'No files given'
        file_ps = sorted(file_ps)  # we assume that smallest shard is at the end
        num_per_shard_prev = None
        ps = []
        for i, p in enumerate(file_ps):
            num_per_shard = get_num_in_shard(p, files_and_shards, opener)
            if num_per_shard_prev is None:  # first file
                num_per_shard_prev = num_per_shard
                ps.append(p)
                continue
            if num_per_shard_prev < num_per_shard:
                raise ValueError('Expected all shards to have the same number of elements,'
                                 'except last one. Previous had {} elements, current ({}) has {}!'.format(
                                    num_per_shard_prev, p, num_per_shard))
            if num_per_shard_prev > num_per_shard:  # assuming this is the last
                is_last = i == len(file_ps) - 1
                if not is_last:
                    raise ValueError(
                            'Found shard with too few elements, and it is not the last one! {}\n'
                            'Last: {}\n'.format(p, file_ps[-1]))
                print('Filtering shard {}, dropping {} elements...'.format(p, num_per_shard))
                break  # is last anyways
            else:  # same numer as before, all good
                ps.append(p)
        return ps, num_per_shard_prev

def get_num_in_shard(shard_p, files_and_shards, opener=default_opener):
    hdf5_root = os.path.dirname(shard_p)
    if files_and_shards[os.path.basename(shard_p)]:
        num_per_shard = files_and_shards[os.path.basename(shard_p)]
    else:
        print('\rOpening {}...'.format(shard_p), end='')
        with opener(shard_p) as f:
            num_per_shard = len(f.keys())
    return num_per_shard

def getitem_func(file, index):
    """
        Return image and labels from MPIIFaceGaze with correct shapes.
        PyTorch needs the image to be [channel x width x height].
        The images came as BGR, so there was a need to invert it
        to RGB also.
    """
    img = np.transpose(file['data'][index][[2, 1, 0], :, :], (1, 2, 0))
    label = file['label'][index]
    return {'sample': img, 'label': label}

def get_loaders(files_and_shards, files_path, test_filenames, custom_getitem,
                extension='.h5', transform=None, transform_label=None,
                batch_size=16, num_workers=8, use_gpu=True):
    """
        Parameters:
            - files_and_shards: dictionary containing entries such as
                    {file1: num_shards1, ..., fileN: num_shardsN}.
            This means it's the filename and the amount of data in it. It's
            expected the last file to have fewer entries, and the rest have
            the same amount of data.
            - files_path: path to folder where files are (in this case, all files
            are in the same folder, not separated in train/test folders).
            - test_filenames: which files are to be separated for validation/test.
            - custom_getitem: custom function defined by user to get data from
            the hdf5 file. It's inputs are be opened hdf5 file and sampled index.
            - extension: file extension for hdf5 file ('.h5', '.hdf5', ...)
            - transform and transform_label: custom data transforms.
            - batch_size: PyTorch dataloader batch_size.
            - num_workers: PyTorch's dataloader num_workers param.
            - use_gpu: PyTorch's dataloader pin_memory param.
    """

    train_files = glob.glob(files_path + '*' + extension)

    test_files = [file for file in train_files if any(f in file for f in test_filenames)]

    train_files = [file for file in train_files if file not in test_files]

    train_dataset = HDF5Dataset(train_files, custom_getitem=getitem_func,
                 files_and_shards=files_and_shards, transform=transform,
                 transform_label=transform_label)

    test_dataset = HDF5Dataset(test_files, custom_getitem=getitem_func,
                 files_and_shards=files_and_shards, transform=transform,
                 transform_label=transform_label)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=use_gpu,
                              drop_last=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=use_gpu,
                             drop_last=False)

    del train_dataset, test_dataset # let's clear all memory we can lol

    return train_loader, test_loader

# train_loader, test_loader = get_loaders(files_and_shards, path, ['p05_0', 'p07_0', 'p12_1'], getitem_func, batch_size=1)
# files_and_shards = {'p{:02}_{}.h5'.format(i, j):1500 for j in range(2) for i in range(15)}

