import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import os

filepath, _ = os.path.split(__file__)
#print(__file__)
if filepath != '':
    path = filepath + '/source/KDtree.so'
else:
    path = 'source/KDtree.so'
kdtree = ctypes.CDLL(path)
SearchRadius = kdtree.SearchForRadius
# float * point, int point_size,
# int * query_id, int query_size,
# float distance, int max_count,
# int * result, int dim = 3
SearchRadius.argtypes = [ndpointer(ctypes.c_float), ctypes.c_int,  # pointes
                         ndpointer(ctypes.c_float), ctypes.c_int,  # query
                         ctypes.c_float, ctypes.c_int,
                         ndpointer(ctypes.c_int), ctypes.c_int]

def query_ball_point_one_KD(points, query_points, radius, max_number, times = 3):
# times: for searching;
    #print('from python', points.shape, points, query_points)
    MAX_NUM = max_number * times
    num_point = points.shape[0] # batch_size
    num_query = query_points.shape[0]
    real_num_query = num_query 
    if query_points.shape[0] & (query_points.shape[0] - 1) != 0:
        value = 1
        while value < num_query:
            value *= 2
        query_points = np.concatenate((query_points, query_points))[:value]
        num_query = value
    #print(points.shape, query_points.shape, num_query)
    #int(points, query_points, num_point, num_query, MAX_NUM)
    points = points.astype(np.float32)
    query_points = query_points.astype(np.float32)
    result = np.zeros((num_query, MAX_NUM)).astype(np.int32)
    SearchRadius(points, num_point,
                 query_points, num_query,
                 radius, MAX_NUM,
                 result, 3)
    result = result[:real_num_query]
    return result

def query_ball_point_KD(points, query_points, radius, max_number, times = 3):
    batch_size = points.shape[0]
    result = np.zeros((query_points.shape[0], query_points.shape[1], max_number)).astype(np.int32)
    for i in range(batch_size):
        now = query_ball_point_one_KD(points[i], query_points[i], radius, max_number, times)
        result[i] = now[:, :max_number]
    return result


def index_points_numpy_for_testing(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    result = np.zeros((points.shape[0], idx.shape[1], points.shape[2]))
    for i in range(points.shape[0]):
        result[i] = points[i, idx[i]]
    return result

    return new_points

if __name__ == '__main__':
    import time
    points = np.random.rand(4096, 3)
    query_points = np.random.randint(0, 4096, 1024)
    print(query_ball_point_one_KD(points, points[query_points], 0.1, 50))
    #exit()
    t = time.time()
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    x = query_ball_point_one_KD(points, points[query_points], 0.1, 50)
    print((time.time()-t)/9)
    t = time.time()
    points = np.random.random((8, 8192, 3))
    query_points = np.random.randint(0, 8192, (8, 8192))
    #print(points, query_points, 0.1, 16)
    now = index_points_numpy_for_testing(points,query_points)
    print('index: ', time.time()-t)
    query_ball_point_KD(points, now, 0.1, 32)
    #print(query_ball_point_KD(points, now, 0.1, 16))
    print(time.time()-t)

