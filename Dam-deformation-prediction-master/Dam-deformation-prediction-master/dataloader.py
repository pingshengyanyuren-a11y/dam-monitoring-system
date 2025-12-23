from __future__ import division
import numpy as np
import torch
import os
import logging
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger('MYAR.Data')

class Data_set(Dataset):
    def __init__(self, data_dir, index, train=True):
        self.data_input = np.load(os.path.join(data_dir, r'train_x_input.npy'))[index]
        self.label = np.load(os.path.join(data_dir, r'train_label.npy'))[index]
        self.len = self.data_input.shape[0]
        if train:
            logger.info(f'train_len: {self.len}')
        else:
            logger.info(f'val_len: {self.len}')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_input[index], self.label[index]


class WeightedSampler(Sampler):
    def __init__(self, data_dir, train_index, replacement=True):
        v = np.load(os.path.join(data_dir, r'train_v.npy'))[train_index]
        self.weights = torch.as_tensor(np.abs(v[:, 0]) / np.sum(np.abs(v[:, 0])), dtype=torch.double)
        logger.info('weights: {}'.format(self.weights))
        self.num_samples = self.weights.shape[0]
        logger.info(r'num samples: {}'.format(self.num_samples))
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples