import torch
import numpy as np
import warnings
import os
import time
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


class ModelNetFeatureDataset(Dataset):
    def __init__(self, root, labelpath, cache_size=15000, build_cache=True, xyz_weight=1):
        self.xyz_weight = xyz_weight
        self.root = root
        self.labelpath = os.path.join(self.root, labelpath)
        self.info = [line.rstrip().split() for line in open(self.labelpath)]
        print('Dataset: The size of data is %d' % (len(self.info)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple
        if build_cache:
            t_start = time.time()
            print('building dataset(during initialize)')
            for index in range(self.__len__()):
                if index % 100 == 0:
                    print('building dataset: index %d, time=%.2f' % (index, time.time()-t_start), flush=True)
                self._get_item(index)


    def __len__(self):
        return len(self.info)

    def _get_item(self, index):
        if index in self.cache:
            sample = self.cache[index]
        else:
            info = self.info[index]
            [cls, datapath] = info
            datapath = os.path.join(self.root, datapath)
            data = [[float(value) for value in line.strip().split()] for line in open(datapath)]
            data = np.array(data)
            data[:, :3] = data[:, :3] * self.xyz_weight
            data = torch.from_numpy(data).float()
            cls = np.array([cls]).astype(np.int32)
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
