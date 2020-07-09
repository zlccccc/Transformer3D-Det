import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import os
import cv2

filepath, _ = os.path.split(__file__)
#print(__file__)
if filepath != '':
    path = filepath + '/source/PointCloudToCylinderMap.so'
else:
    path = 'source/PointCloudToCylinderMap.so'
CylinderMap = ctypes.CDLL(path)
CylinderMap_init = CylinderMap.init
H, W = 256, 256
CylinderMap_init(H, W)
CylinderMap_build = CylinderMap.PointCloudToCylinderMap
# float* point, float *feature, int point_size, 
# float *depth_map, float *result, int feature_dim, float *ids
# float *landmarks, int n_landmark, float *landmark_ids
CylinderMap_build.argtypes = [ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ctypes.c_int,  # pointes
                              ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float),
                              ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float)]

def save_labeled_img(image, landmark_pred, landmark_targ, save_base, save_name, norm_back=2):
    if not os.path.exists(save_base):
        os.makedirs(save_base) 
    save_path = os.path.join(save_base, save_name)
    if norm_back:
        image = image / norm_back * 255
    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))
    else:
        pass
    print('image.shape', image.shape, len(image.shape))
    assert len(image.shape) ==3, 'image.shape wrong'
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(int)
    assert image.shape[-1] == 3, 'image.shape[-1](rgb) != 3'
    #print(image.shape, np.max(image))
    for i in range(len(landmark_targ)):
        targ_pt = (round(float(landmark_targ[i][0])), round(float(landmark_targ[i][1])))
        image = cv2.circle(image, targ_pt, 1, (0, 255, 0), 2)
    for i in range(len(landmark_pred)):
        pred_pt = (round(float(landmark_pred[i][0])), round(float(landmark_pred[i][1])))
        image = cv2.circle(image, pred_pt, 1, (0, 0, 255), 2)
    image = cv2.flip(image, 0)
    cv2.imwrite(save_path, image)

def getCylinderMap(points, features, landmarks): #n*3, n*k
    #print('from python', points.shape, points, query_points)
    num_point, dim = features.shape
    points = points.astype(np.float32)
    features = features.astype(np.float32)
    landmarks = landmarks.astype(np.float32)
    depth_map = np.zeros((H, W)).astype(np.float32)
    result = np.zeros((H, W, features.shape[-1])).astype(np.float32)
    ids = np.zeros((num_point, 2)).astype(np.float32)
    id_landmarks = np.zeros((landmarks.shape[0], 2)).astype(np.float32)
    lm_size = landmarks.shape[0]
    #print('lm_size', lm_size)
    #print('before build', landmarks)
    #print(features.shape, points.shape, depth_map.shape, result.shape, dim, ids.shape)
    CylinderMap_build(points, features, num_point,
                      depth_map, result, dim, ids,
                      landmarks, lm_size, id_landmarks)
    if True:
        depth_map = cv2.bilateralFilter(depth_map, 5, 10, 5)
        features = cv2.bilateralFilter(features, 5, 10, 5)
    return depth_map, result, ids, id_landmarks


def getBatchCylinderMap(points, features, landmarks):
    batch_size = points.shape[0]
    depth_maps = np.zeros((batch_size, H, W)).astype(np.float32)
    results = np.zeros((batch_size, H, W, features.shape[-1])).astype(np.float32)
    ids = np.zeros((batch_size, points.shape[1], 2)).astype(np.float32)
    landmarks = landmarks.reshape(landmarks.shape[0], -1, 3)
    id_landmarks = np.zeros((batch_size, landmarks.shape[1], 2)).astype(np.float32)
    for i in range(batch_size):
        depth_maps[i], results[i], ids[i], id_landmarks[i] = getCylinderMap(points[i], features[i], landmarks[i])
    return depth_maps, results, ids, id_landmarks


if __name__ == '__main__':
    import time
    t = time.time()
    points = np.random.random((8192, 3)) * 2 - 1
    features = np.random.random((8192, 3)) * 2 - 1
    lm = np.random.random((12, 3))
    #print(points, query_points, 0.1, 16)
    print('index: ', time.time()-t)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    #p_cat = np.hstack((points, ids_3Dto2D))
    #print(ids_3Dto2D)
    print(depthmap / 2 * 255)
    print(np.max(depthmap))
    print(time.time()-t)
    save_labeled_img(depthmap, [], [], 'testing_pictures', 'depth_map.jpg')
    #print(depthmap, featuremap, ids_3Dto2D)
    t = time.time()
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    depthmap, featuremap, ids_3Dto2D, lm_pos = getCylinderMap(points, points, lm)
    print(time.time()-t)

