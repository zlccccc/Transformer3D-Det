import torch
import numpy as np
from torch.utils.data import Dataset


class Test3DDataset(Dataset):
    def __init__(self, npoint, split, transform=None):
        super(Test3DDataset, self).__init__()
        self.npoint = npoint
        self.data = [i for i in range(350)]
        self.transform = transform
        if split == 'train':
            self.data = self.data[:250]
        elif split == 'validate':
            self.data = self.data[250:300]
        elif split == 'test':
            self.data = self.data[300:350]
        else:
            raise NotImplementedError(split)

    def __getitem__(self, index):
        # print('dataset: testitem %d' % index)
        sample = {}
        point_set = np.random.normal(size=(self.npoint, 3))
        seg = np.random.randint(0, 3, size=(self.npoint, 1))
        cls_ = np.random.randint(0, 3, size=(1))
        sample['point_set'] = torch.from_numpy(point_set).float()
        sample['cls'] = torch.from_numpy(cls_).long()
        sample['seg'] = torch.from_numpy(seg).float()
        return sample

    def __len__(self):
        return len(self.data)
