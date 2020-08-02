########### DataLoader for scene graph generation using point cloud.  ###############################

import numpy as np
import warnings
import os
from torch.utils.data import Dataset

import glob
import json
import copy

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def random_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class PCSGDataset(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=False, cache_size=0, object_only=False, rel_only=False, debug=False):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform

        self.normal_channel = normal_channel
        self.data_split = split
        self.object_only = object_only
        self.rel_only = rel_only
        ######## scene graph version ##########################
        train_path = os.path.join(root, 'train')
        # if (debug == True):
        #     train_path = '/home/zzha5029/datasets/3DSSG/test3/train/'
        train_files = [f for f in glob.glob(train_path + "/*.json", recursive=True)]

        val_path = os.path.join(root, 'val')
        # if (debug == True):
        #     val_path = '/home/zzha5029/datasets/3DSSG/test3/val/'
        val_files = [f for f in glob.glob(val_path + "/*.json", recursive=True)]

        print(train_files[0])
        print(val_files[0])

        assert (split == 'train' or split == 'val')
        assert (self.normal_channel == False)
        if (split == 'train'):
            self.datapath = train_files
            #f = open('/home/zzha5029/datasets/3DSSG/3DSSG_subset/relationships_train.json')
            #gt_data = json.load(f)
        elif(split == 'val'):
            self.datapath = val_files
            #f = open('/home/zzha5029/datasets/3DSSG/3DSSG_subset/relationships_validation.json')
            #gt_data = json.load(f)
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple  TODO get cache
        print('------------------------------------')

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        # print(fn)
        f = open(fn)
        sg_data = json.load(f)
        # print(sg_data)
        if (self.rel_only == False):  # at most 9 objs
            object_keys = list(sg_data['object'].keys())
            # print(object_keys)
            object_num = len(object_keys)
            #print('check point object num in Dataloader')
            # print(object_num)
            object_point_set = np.zeros((9, self.npoints, 3))
            object_cls = np.zeros((9, 1))
            object_idx = np.zeros((9, 1))
            multi_batch_object_mask = np.zeros((1))
            multi_batch_object_mask = object_num
            # print('chcek point object mask in Dataloader')
            # print(multi_batch_object_mask)
            for i in range(0, object_num):
                object_key = object_keys[i]
                _, object_id = object_key.split('_')
                object_idx[i] = int(object_id)
                # print(object_id)
                data_root = '/home/zzha5029/datasets/3DSSG/lighter_point_cloud/' + self.data_split + '/'
                #data_root = '/home/zzha5029/datasets/3DSSG/test2/' + self.data_split + '/'
                object_pc_data_dir = data_root + object_key + '.npy'
                object_pc_data = np.load(object_pc_data_dir)
                object_pc_label = sg_data['object'][object_key]
                object_pc_data_num = len(object_pc_data)
                # print(object_pc_data_num)
                #print(object_pc_data.shape, object_pc_label)
                random_index = np.random.randint(object_pc_data_num, size=self.npoints)
                sampled_object_pc_data = object_pc_data[random_index]
                # print(random_index[-1])
                #print(sampled_object_pc_data.shape, object_pc_label)
                # print(object_point_set.shape)
                object_point_set[i, :, :] = copy.copy(sampled_object_pc_data)
                object_cls[i] = copy.copy(object_pc_label)

        if (self.object_only == False):  # at most 72 objs
            rel_keys = list(sg_data['rel'].keys())
            rel_num = len(rel_keys)
            # print(rel_num)
            rel_point_set = np.zeros((72, self.npoints, 4))  # maximum 
            rel_cls_one_hot = np.zeros((72, 27))
            rel_mask = np.ones((72, 1))
            rel_idx = np.zeros((72, 2))
            multi_batch_rel_mask = rel_num
            #print('multi_batch_rel_mask in Dataloader')
            # print(multi_batch_rel_mask)
            # print(rel_keys[0])
            for i in range(0, rel_num):
                rel_key = rel_keys[i]
                _, subject_id, object_id = rel_key.split('_')
                rel_idx[i, 0] = int(subject_id)
                rel_idx[i, 1] = int(object_id)
                #print(subject_id, object_id)
                data_root = '/home/zzha5029/datasets/3DSSG/lighter_point_cloud/' + self.data_split + '/'
                #data_root = '/home/zzha5029/datasets/3DSSG/test2/' + self.data_split + '/'
                rel_pc_data_dir = data_root + rel_key + '.npy'
                rel_pc_data = np.load(rel_pc_data_dir)
                # print(rel_pc_data.dtype)
                rel_pc_label = sg_data['rel'][rel_key]
                rel_pc_data_num = len(rel_pc_data)
                # print(rel_pc_data_num)
                if (rel_pc_label != None):
                    for ele in rel_pc_label:
                        rel_cls_one_hot[i, ele] = 1
                else:
                    rel_cls_one_hot[i, 0] = 1
                    rel_mask[i, 0] = 0
                #print(rel_pc_data.shape, rel_pc_label)
                random_index = np.random.randint(rel_pc_data_num, size=self.npoints)
                sampled_rel_pc_data = rel_pc_data[random_index]
                # print(random_index[-1])
                #print(sampled_rel_pc_data[:,:3].shape, rel_pc_label)
                rel_point_set[i, :, :] = copy.copy(sampled_rel_pc_data[:, :])
                #print(rel_pc_data.shape, rel_pc_label)
                # print(rel_cls_one_hot[i,:])

        sample = {}
        if not self.object_only:
            sample['rel_point_set'] = rel_point_set
            sample['rel_cls_one_hot'] = rel_cls_one_hot
            sample['rel_idx'] = rel_idx
            sample['rel_mask'] = rel_mask
            sample['multi_batch_rel_mask'] = multi_batch_rel_mask
        if not self.rel_only:
            sample['object_point_set'] = object_point_set
            sample['object_cls'] = object_cls
            sample['object_idx'] = object_idx
            sample['multi_batch_object_mask'] = multi_batch_object_mask
        return sample

    def __getitem__(self, index):
        return self._get_item(index)
