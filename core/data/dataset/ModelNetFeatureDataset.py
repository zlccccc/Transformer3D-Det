import torch
import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


class ModelNetFeatureDataset(Dataset):
    def __init__(self, root, labelpath, cache_size=15000):
        self.root = root
        self.labelpath = os.path.join(self.root, labelpath)
        self.info = [line.rstrip().split() for line in open(self.labelpath)]
        print('Dataset: The size of data is %d' % (len(self.info)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.info)

    def _get_item(self, index):
        if index in self.cache:
            sample = self.cache[index]
        else:
            info = self.info[index]
            [cls, datapath] = info
            datapath = os.path.join(self.root, datapath)
            data = [[float(value) for value in line.strip()] for line in open(datapath)]
            data = np.array(data)
            data = torch.from_numpy(data).float()
            sample = {}
            sample['point_set'] = data
            sample['cls'] = torch.from_numpy(cls).long()
            if len(self.cache) < self.cache_size:
                self.cache[index] = sample
        return sample

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    data = ModelNetFeatureDataset('/data/modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
