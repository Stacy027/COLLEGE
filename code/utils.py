from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
from itertools import chain

class OTFDistributedSampler(Sampler):
    # On-The-Fly sampler
    def __init__(self, indexed_train_fps, n_workers, rank, shuffle=True):
        super(OTFDistributedSampler, self).__init__(0)
        self.epoch = 0
        self.shuffle = shuffle

        file_per_process = len(indexed_train_fps) // n_workers
        if file_per_process * n_workers != len(indexed_train_fps):
            if rank == 0:
                print('[Sampler] Drop {} files.'.format(len(indexed_train_fps) - file_per_process * n_workers))
                print('[Sampler] # files per process: {}'.format(file_per_process))
        self.fps = indexed_train_fps[rank * file_per_process:(rank + 1) * file_per_process]
        self.file_per_process = file_per_process

        data = []
        with open(self.fps[0], 'r', encoding='utf-8') as fin:
            import json
            for x in fin:
                data.append(json.loads(x))
            self.num_samples_per_file = len(data)
        assert self.num_samples_per_file == 5000
        self.total_num_samples = self.num_samples_per_file * len(self.fps)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = []
            for i in range(self.file_per_process):
                indexes = list(np.arange(self.num_samples_per_file) + i * self.num_samples_per_file) # indices within one file
                np.random.shuffle(indexes)
                indices.append(indexes)
            np.random.shuffle(indices)
            indices = list(chain(*indices))
        else:
            indices = []
            for i in range(self.file_per_process):
                indexes = list(np.arange(self.num_samples_per_file) + i * self.num_samples_per_file)
                indices.extend(indexes)

        return iter(indices)

    def __len__(self):
        return self.total_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

