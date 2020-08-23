from core.model.RandLANet.utils.helper_tool import DataProcessing as DP
from core.model.RandLANet.utils.helper_tool import ConfigSemantic3D as cfg
from .utils.helper_ply import read_ply
from os.path import join, exists
import numpy as np
import os
import pickle
import torch.utils.data as torch_data
import torch


class Semantic3DDataset(torch_data.Dataset):
    def __init__(self, mode, data_path):
        self.name = 'Semantic3D'
        self.path = data_path
        self.label_to_names = {0: 'unlabeled',
                               1: 'man-made terrain',
                               2: 'natural terrain',
                               3: 'high vegetation',
                               4: 'low vegetation',
                               5: 'buildings',
                               6: 'hard scape',
                               7: 'scanning artefacts',
                               8: 'cars'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        # Following KPConv to do the train-validation split
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = 1

        # Initial training-validation-testing files (filename)
        train_files = []
        val_files = []
        test_files = []
        cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == '.txt']
        for pc_name in cloud_names:
            if exists(join(self.original_folder, pc_name + '.labels')):
                train_files.append(join(self.sub_pc_folder, pc_name + '.ply'))
            else:
                test_files.append(join(self.full_pc_folder, pc_name + '.ply'))

        tmp_training_files = train_files
        train_files = []
        for i, file_path in enumerate(tmp_training_files):
            if self.all_splits[i] == self.val_split:
                val_files.append(file_path)
            else:
                train_files.append(file_path)

        self.mode = mode
        if mode == 'training':
            self.data_list = train_files
        elif mode == 'validation':
            self.data_list = val_files
        elif mode == 'test':
            self.data_list = test_files
        else:
            raise NotImplementedError(mode)
        self.data_list = DP.shuffle_list(self.data_list)

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Ascii files dict; JUST for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        self.init_batch_gen(mode)

    def load_sub_sampled_clouds(self, sub_grid_size):  # LOAD ALL FOR GENERATE

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = self.data_list

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            cloud_split = self.mode

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            if cloud_split == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if file_path in self.val_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            if file_path in self.test_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        print('finished')
        return

    def init_batch_gen(self):
        split = self.mode
        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize (for every point)
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

    def spatially_regular_gen(self):
        split = self.mode
        # Generator loop; this func Can generate inf times
        # Choose the cloud with the lowest probability
        cloud_idx = int(np.argmin(self.min_possibility[split]))
        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[split][cloud_idx])
        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[split][cloud_idx].data, copy=False)
        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)
        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
        # Shuffle index
        query_idx = DP.shuffle_idx(query_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[query_idx]
        queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
        queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
        if split == 'test':
            queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
            queried_pt_weight = 1
        else:
            queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
            queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
            queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])
        # Update the possibility of the selected points
        dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
        self.possibility[split][cloud_idx][query_idx] += delta
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
        return (queried_pc_xyz,
                queried_pc_colors.astype(np.float32),
                queried_pc_labels,
                query_idx.astype(np.int32),
                np.array([cloud_idx], dtype=np.int32))

    def downsample_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
        batch_features = self.torch_augment_input(batch_xyz, batch_features)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neigh_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neigh_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_xyz, 1)
            input_points.append(batch_xyz)
            input_neighbors.append(neigh_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

        return input_list

    # data augmentation
    @staticmethod
    def torch_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = torch.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = torch.cos(theta), torch.sin(theta)
        cs0 = torch.zeros_like(c)
        cs1 = torch.ones_like(c)
        R = torch.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = torch.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = torch.reshape(torch.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = torch.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = torch.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(torch.round(torch.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(torch.ones([1, 1], dtype=torch.float32))
        s *= torch.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = torch.tile(s, [torch.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = torch.random_normal(torch.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        rgb = features[:, :3]
        stacked_features = torch.concat([transformed_xyz, rgb], axis=-1)
        return stacked_features

    def __len__(self):
        return self.num_per_epoch  # iter per epoch

    def __getitem__(self, item):
        # queried_pc_xyz,
        # queried_pc_colors.astype(np.float32),
        # queried_pc_labels,
        # query_idx.astype(np.int32),
        # np.array([cloud_idx])
        selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item)
        return selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind

    def collate_fn(self, batch):  # MUST DO IT; AS KD is BATCHWISE
        selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind = [], [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_colors.append(batch[i][1])
            selected_labels.append(batch[i][2])
            selected_idx.append(batch[i][3])
            cloud_ind.append(batch[i][4])

        selected_pc = np.stack(selected_pc)
        selected_colors = np.stack(selected_labels)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.downsample_map(selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()
        return inputs
