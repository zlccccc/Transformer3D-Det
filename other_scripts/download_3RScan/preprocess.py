import json
from plyfile import PlyData, PlyElement
import numpy as np
import time
import copy
import os
import numpy as np


def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                 max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] >= min_x, points[:, 0] <= max_x)
    bound_y = np.logical_and(points[:, 1] >= min_y, points[:, 1] <= max_y)
    bound_z = np.logical_and(points[:, 2] >= min_z, points[:, 2] <= max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter


def label_rel_point_cloud(points_inside_box, selected_point_clouds, subject_unit_id, object_unit_id):

    a = copy.copy(np.array(points_inside_box))
    b = np.array(selected_point_clouds[subject_unit_id])
    c = np.array(selected_point_clouds[object_unit_id])

    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    points_inside_box_num = len(a)

    all_rel_point_cloud = copy.copy(np.array(points_inside_box))
    subject_rel_point_cloud = copy.copy(np.array(selected_point_clouds[subject_unit_id]))
    object_rel_point_cloud = copy.copy(np.array(selected_point_clouds[object_unit_id]))
    points_inside_box_num = len(all_rel_point_cloud)
    other_rel_point_cloud = []

    for i in range(0, len(all_rel_point_cloud)):
        if (str(int(all_rel_point_cloud[i, 6])) != subject_unit_id) and (str(int(all_rel_point_cloud[i, 6])) != object_unit_id):
            other_rel_point_cloud.append(all_rel_point_cloud[i, :3])

    other_rel_point_cloud = np.array(other_rel_point_cloud)

    if len(other_rel_point_cloud) > 0:
        a_point_cloud = np.zeros((len(other_rel_point_cloud), 4))
    b_point_cloud = np.ones((len(object_rel_point_cloud), 4))
    c_point_cloud = 2 * np.ones((len(subject_rel_point_cloud), 4))

    if len(other_rel_point_cloud) > 0:
        a_point_cloud[:, :3] = other_rel_point_cloud
    b_point_cloud[:, :3] = object_rel_point_cloud
    c_point_cloud[:, :3] = subject_rel_point_cloud

    if len(other_rel_point_cloud) > 0:
        labeled_rel_point_cloud = np.concatenate((a_point_cloud, b_point_cloud, c_point_cloud), axis=0)
    else:
        labeled_rel_point_cloud = np.concatenate((b_point_cloud, c_point_cloud), axis=0)

    if points_inside_box_num != len(labeled_rel_point_cloud):
        print('error!!!!!')
        # print(points_inside_box_num)
        # print(len(labeled_rel_point_cloud))

    return labeled_rel_point_cloud


root = '/data1/zhaolichen/data'
split = 'train'
# split = 'validation'

save_root_dir = root + '/3DSSG/cropped_point_cloud/'
if not os.path.exists(save_root_dir):
    os.makedirs(save_root_dir)

# Load relationship JSON file
f = open(root + '/3DSSG/3DSSG_subset/relationships_%s.json' % split)
data = json.load(f)
# print(data.keys())
# print the number of graph
train_subgraph_num = len(data['scans'])
print(train_subgraph_num)

#graph_id = 0

# print(data['scans'][graph_id].keys())
# print('--------------------------------------------------------------')
# for i in range(0,50):
#    print(data['scans'][i]['scan'])
# print('--------------------------------------------------------------')
# print(data['scans'][graph_id]['scan'])
#print('object is ...')
# print(data['scans'][graph_id]['objects'])
# print(len(data['scans'][graph_id]['objects']))

# get the number of objects and relationships in each graph
for graph_id in range(0, train_subgraph_num):
    print('process the No.' + str(graph_id) + ' subgraph.')

    rel_len = len(data['scans'][graph_id]['relationships'])
    object_ids = list(data['scans'][graph_id]['objects'].keys())
    objects_num = len(object_ids)
    # print(object_ids)
    # print('---')
    # for i in range(0, rel_len):
    # print(data['scans'][0]['relationships'][i])

    # load point cloud data
    root_dir = root + '/3RScan/' + data['scans'][graph_id]['scan'] + '/'
    plypath = root_dir + 'labels.instances.annotated.ply'
    print(plypath)
    if not os.path.exists(plypath):
        print('path %s not exist!', plypath)
        continue
    plydata = PlyData.read(root_dir + 'labels.instances.annotated.ply')
    # print(plydata.elements[0].properties)
    point_cloud_num = len(plydata.elements[0]['objectId'])
    scene_point_cloud = np.zeros((point_cloud_num, 7))
    scene_point_cloud[:, 0] = plydata.elements[0]['x']
    scene_point_cloud[:, 1] = plydata.elements[0]['y']
    scene_point_cloud[:, 2] = plydata.elements[0]['z']
    scene_point_cloud[:, 3] = plydata.elements[0]['red']
    scene_point_cloud[:, 4] = plydata.elements[0]['green']
    scene_point_cloud[:, 5] = plydata.elements[0]['blue']
    scene_point_cloud[:, 6] = plydata.elements[0]['objectId']
    # print(len(plydata.elements[0]['objectId']))
    # print(len(plydata.elements[0]['x']))
    # print(len(plydata.elements[0]['y']))
    # print(len(plydata.elements[0]['z']))

    # divide the point cloud to each object
    selected_point_clouds = {}
    objects_bbox = {}
    objects_point_clouds = {}

    for selected_pc_id in object_ids:
        selected_point_clouds[selected_pc_id] = []
        objects_bbox[selected_pc_id] = []

    # divide each point to their objectID
    for i in range(0, point_cloud_num):
        pc_objectID = str(plydata.elements[0]['objectId'][i])
        if pc_objectID in object_ids:
            x = plydata.elements[0]['x'][i]
            y = plydata.elements[0]['y'][i]
            z = plydata.elements[0]['z'][i]
            selected_point_clouds[pc_objectID].append([x, y, z])

    # calculate the bounding box for each object
    real_pc_ids = []
    for selected_pc_id in object_ids:
        # print(bbox)
        point_cloud_data = np.array(selected_point_clouds[selected_pc_id])
        # print(point_cloud_data.shape)
        if len(point_cloud_data) == 0: 
           continue
        real_pc_ids.append(selected_pc_id)
        object_name = data['scans'][graph_id]['objects'][selected_pc_id]
        bbox = np.zeros((3, 2))
        max_value = np.max(point_cloud_data, axis=0)
        # print(max_value.shape)
        min_value = np.min(point_cloud_data, axis=0)
        # print(min_value.shape)
        bbox[:, 0] = min_value
        bbox[:, 1] = max_value
        objects_bbox[selected_pc_id] = bbox
        # print(point_cloud_data.shape)
        key = selected_pc_id.zfill(3)
        # print(key)
        objects_point_clouds[key] = copy.copy(point_cloud_data)

        # check the output is correct or not
        #file_name = str(graph_id).zfill(4) + '_obj_' + selected_pc_id.zfill(3) +'_'+ object_name + '.xyz'
        # print(file_name)
        #pred_pc = point_cloud_data[:,:3]
        # np.savetxt(save_root_dir+file_name,pred_pc,fmt='%.6f')
    object_ids = real_pc_ids

    rel_point_clouds = {}
    start_time = time.time()
    # calculate the bounding box for a pair of objects
    for subject_unit_id in object_ids:
        for object_unit_id in object_ids:
            if (object_unit_id != subject_unit_id):
                object_name = data['scans'][graph_id]['objects'][object_unit_id]
                subject_name = data['scans'][graph_id]['objects'][subject_unit_id]

                subject_unit_bbox = objects_bbox[subject_unit_id]
                object_unit_bbox = objects_bbox[object_unit_id]
                bbox = np.array([subject_unit_bbox, object_unit_bbox])
                max_value = np.max(bbox[:, :, 1], axis=0)
                min_value = np.min(bbox[:, :, 0], axis=0)
                min_x, min_y, min_z = min_value
                max_x, max_y, max_z = max_value
                inside_box = bounding_box(scene_point_cloud, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, min_z=min_z, max_z=max_z)
                points_inside_box = scene_point_cloud[inside_box]

                # label the object, subject and others
                labeled_rel_point_cloud = label_rel_point_cloud(points_inside_box, selected_point_clouds, subject_unit_id, object_unit_id)
                # print(labeled_rel_point_cloud.shape)
                key = subject_unit_id.zfill(3) + '_' + object_unit_id.zfill(3)
                # print(key)
                rel_point_clouds[key] = copy.copy(labeled_rel_point_cloud)
                # print('------------------------------------')

                # check the output is correct or not
                #file_name = str(graph_id).zfill(4) + '_rel_' + object_name +'_'+ subject_name + '.xyz'
                #pred_pc = labeled_rel_point_cloud[:,:3]
                # np.savetxt(save_root_dir+file_name,pred_pc,fmt='%.6f')
    duration = time.time() - start_time
    print(duration)

    # save as json file
    subgraph_all = {}
    subgraph_all['object'] = objects_point_clouds
    subgraph_all['rel'] = rel_point_clouds

    file_name = save_root_dir + split + '/'
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    file_name = file_name + str(graph_id).zfill(4) + '.json'
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)
