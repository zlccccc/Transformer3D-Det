# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pc_util
import numpy as np
import sys
import os


def save_obj(input_folder, scene_name, output_folder):
    # scene_name = 'scannet_train_detection_data/scene0002_00'
    # output_folder = 'data_viz_dump'
    scene_name_all = os.path.join(input_folder, scene_name)
    data = np.load(scene_name_all+'_vert.npy')
    scene_points = data[:, 0:3]
    colors = data[:, 3:]
    instance_labels = np.load(scene_name_all+'_ins_label.npy')
    semantic_labels = np.load(scene_name_all+'_sem_label.npy')
    instance_bboxes = np.load(scene_name_all+'_bbox.npy')

    print(scene_name, '    <<< save obj: obj instance & semantic labels', flush=True)
    print(np.unique(instance_labels))
    print(np.unique(semantic_labels))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Write scene as OBJ file for visualization
    pc_util.write_ply_rgb(scene_points, colors, os.path.join(output_folder, '%s_scene.obj' % scene_name))
    pc_util.write_ply_color(scene_points, instance_labels, os.path.join(output_folder, '%s_scene_instance.obj' % scene_name))
    pc_util.write_ply_color(scene_points, semantic_labels, os.path.join(output_folder, '%s_scene_semantic.obj' % scene_name))

    print('save obj at', output_folder)
    print(instance_bboxes.shape, '<< instance boxes.shape')
    # input()
    return instance_bboxes.shape[0]
