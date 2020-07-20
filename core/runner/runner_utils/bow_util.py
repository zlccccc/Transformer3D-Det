import numpy as np
import torch

# __all__ = ['initialize_centers', 'compute_centers', 'compute_bow']


def initialize_centers(num_centers, num_channel=3, normalize=True):
    # center = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(a, b)))
    center = torch.FloatTensor(num_centers, num_channel).uniform_(-1, 1)
    if normalize:
        magnitude = torch.sum(center ** 2, dim=1)
        magnitude = torch.sqrt(magnitude)
        status = magnitude > 1
        status = status.type(torch.FloatTensor)
        factor = (1. - status) + status / (magnitude + 1e-5)
        factor = factor.unsqueeze(-1)
        center *= factor
    return center


# def initialize_centers(num_centers, normalize=True):
#     '''
#     Return:
#         centers: shape (num_centers, 3)
#     '''
#     centers = tf.random.uniform(shape=(num_centers, 3), minval=-1, maxval=1, dtype=tf.float32)
#     if normalize:
#         # normalize to within a unit ball
#         magnitude = tf.reduce_sum(centers ** 2, axis=-1)    # shape (num_centers, 1)
#         magnitude = tf.sqrt(magnitude)
#
#         status = tf.cast(tf.greater(magnitude, 1), tf.float32)
#         factor = (1. - status) + status / (magnitude + 1e-3)
#         factor = tf.expand_dims(factor, axis=-1)
#         centers *= factor
#
#     return centers

def compute_bow(pred):
    '''
    Inputs:
        pred: shape (batch_size, num_points, num_centers)
    Return:
        histogram: shape (batch_size, num_centers)
    '''
    histogram = torch.sum(pred, dim=1)

    return histogram

# def compute_bow(pred):
#     '''
#     Inputs:
#         pred: shape (batch_size, num_points, num_centers)
#     Return:
#         histogram: shape (batch_size, num_centers)
#     '''
#     histogram = tf.reduce_sum(pred, axis=1)
#
#     return histogram


def compute_centers(samples, centers, feature_weight=None):
    batchsize, num_points = samples.size(0), samples.size(1)
    num_centers = centers.size(0)
    samples_ex = torch.unsqueeze(samples, 2)
    samples_ex = samples_ex.repeat(1, 1, num_centers, 1)
    centers_ex = centers.unsqueeze(0).unsqueeze(0)
    centers_ex = centers_ex.repeat(batchsize, num_points, 1, 1)
    # import pdb
    # pdb.set_trace()
    leng = (samples_ex - centers_ex) ** 2
    if feature_weight is not None:
        feature_weight = feature_weight.unsqueeze(2)
        # print(leng.shape, feature_weight.shape)
        feature_weight = feature_weight.repeat(1, 1, num_centers, 1)
        leng = torch.mul(leng, feature_weight)
    diff = torch.sum(leng, dim=-1)
    # print(torch.min(diff[0], dim=-1), flush=True)
    dist, pred = torch.min(diff, dim=-1)
    dist_center, pred_center = torch.min(diff, dim=-2)
    
    #print(torch.mean(dist), flush=True)
    # pred = pred.type(torch.FloatTensor)
    pred_onehot = torch.FloatTensor(batchsize, num_points, num_centers).cuda()
    pred_onehot.zero_()
    pred_onehot.scatter_(2, torch.unsqueeze(pred, 2), 1)
    pred_ex = pred_onehot.unsqueeze(-1)
    histogram = torch.sum(pred_onehot, dim=1)
    sum_by_center = torch.sum(samples_ex * pred_ex, dim=[0, 1])
    count_by_center = torch.sum(pred_ex, dim=[0, 1, 3]).unsqueeze(-1)

    return histogram, sum_by_center, count_by_center, torch.mean(torch.sqrt(dist))

#
# def compute_centers(samples, centers):
#     '''
#     Inputs:
#         samples: shape (batch_size, num_points, 3)
#         centers: shape (num_centers, 3)
#     Return:
#         pred: shape (batch_size, num_points, num_centers)
#         sum_by_center: shape (num_centers, 3)
#         count_by_center: shape (num_centers, 1)
#     '''
#     batch_size = samples.get_shape()[0].value
#     num_points = samples.get_shape()[1].value
#     num_centers = centers.get_shape()[0].value
#
#     samples_ex = tf.expand_dims(samples, axis=2)
#         # shape (batch_size, num_points, 1, 3)
#     samples_ex = tf.tile(samples_ex, multiples=(1, 1, num_centers, 1))
#         # shape (batch_size, num_points, num_centers, 3)
#
#     centers_ex = tf.expand_dims(tf.expand_dims(centers, axis=0), axis=0)
#         # shape (1, 1, num_centers, 3)
#     centers_ex = tf.tile(centers_ex, multiples=(batch_size, num_points, 1, 1))
#         # shape (batch_size, num_points, num_centers, 3)
#
#     diff = tf.reduce_sum((samples_ex - centers_ex) ** 2, axis=-1)
#         # shape (batch_size, num_points, num_centers))
#     pred = tf.argmin(diff, axis=-1, output_type=tf.int32)
#         # shape (batch_size, num_points)
#     pred = tf.one_hot(pred, depth=num_centers, axis=-1, dtype=tf.float32)
#         # shape (batch_size, num_points, num_centers)
#     pred_ex = tf.expand_dims(pred, axis=-1)
#         # shape (batch_size, num_points, num_centers, 1)
#     histogram = tf.reduce_sum(pred, axis=1)
#         # shape (batch_size, num_centers)
#
#     sum_by_center = tf.reduce_sum(samples_ex * pred_ex, axis=(0,1))
#         # shape (num_centers, 3)
#     count_by_center = tf.expand_dims(tf.reduce_sum(pred_ex, axis=(0,1,3)), axis=-1)
#         # shape (num_centers, 1)
#
#     return histogram, sum_by_center, count_by_center
#
#
# def runtest():
#     import  data_filter
#     sess = tf.Session()
#
#     train_files, _ = data_filter.select_data_set('modelnet', 'modelnet')
#     data, labels = data_filter.load_all(train_files)
#     data, labels = data_filter.recollect(data, labels, 20)
#
#     batch_size = 32
#     num_points = 1024
#     num_centers = 10
#     num_epochs = 100
#
#     data, labels = data[:,0:num_points,:], np.squeeze(labels)
#     num_batches = data.shape[0] // batch_size
#
#     inputs_ph = tf.placeholder(name='inputs', dtype=tf.float32, shape=(batch_size, num_points, 3))
#     centers_ph = tf.placeholder(name='centers', dtype=tf.float32, shape=(num_centers, 3))
#
#     centers_init = initialize_centers(num_centers)
#     sum_by_centers, count_by_centers = compute_centers(inputs_ph, centers_ph)
#
#     centers_val_old = sess.run(centers_init)
#     centers_val_new = centers_val_old.copy()
#     print('initial centers:\n{}'.format(centers_val_old))
#
#     for epoch in range(num_epochs):
#         data, _, _ = data_filter.provider.shuffle_data(data, labels)
#         total_sum_centers, total_count_centers = 0, 0
#
#         for ix in range(num_batches):
#             batch_inds = np.arange(ix*batch_size, (ix+1)*batch_size)
#             batch_data = data[batch_inds,...]
#             sc_val, cc_val = sess.run(fetches=[sum_by_centers, count_by_centers],
#                 feed_dict={inputs_ph: batch_data, centers_ph: centers_val_new})
#
#             total_sum_centers += sc_val
#             total_count_centers += cc_val
#
#         centers_val_new = total_sum_centers / (total_count_centers + 1e-5)
#         error_val = np.sum((centers_val_new - centers_val_old) ** 2)
#         centers_val_old = centers_val_new.copy()
#
#         print('iteration {}: {:.6f}\n{}'.format(epoch, error_val, centers_val_new))
#
#
# if __name__ == '__main__':
#     runtest()
