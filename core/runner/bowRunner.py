from core.utils.utils import save_checkpoint
from .runner_utils.testRunnerUtils import testmodel
import torch
import time
import traceback
from core.runner.runner_utils.bow_util import initialize_centers, compute_centers
from .iterRunner import iterRunner
from tqdm import tqdm


def bowRunner(info):
    config = info['config']
    model = info['model']
    loggers = info['loggers']
    last_iter = info['last_iter']
    T_START = time.time()
    trainDataLoader = info['traindataloader']
    if last_iter == -1:
        centers_val = initialize_centers(config.num_centers, config.num_channel).cuda()
        for epoch in range(config.epoch_build_dict + 1):
            total_sum_centers, total_count_centers = 0, 0
            # t = time.time()
            total_dist_min, total_count = 0, 0
            for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                points = data['point_set'].float().cuda()
                # print('load', time.time() - t)
                # t = time.time()
                total_count += len(points)
                with torch.no_grad():  # not useful
                    if 'dist_weight' in data.keys():
                        weight = data['dist_weight'].float().cuda()
                        _, sc_val, cc_val, dist_min = compute_centers(points, centers_val, weight)
                    else:
                        _, sc_val, cc_val, dist_min = compute_centers(points, centers_val)
                    total_sum_centers += sc_val
                    total_count_centers += cc_val
                    total_dist_min += dist_min
                # print(time.time() - t)
            next_val = total_sum_centers / total_count_centers
            upd_pos = total_count_centers!=0
            #print(upd_pos.shape, upd_pos.view(-1).shape)
            #print(upd_pos)
            len_count = len(upd_pos[upd_pos])
            upd_pos = upd_pos.repeat(1, total_sum_centers.shape[1])
            centers_val[upd_pos] = next_val[upd_pos]
            if epoch % config.log_freq == 0:
                loggers.update_loss({'info_out': 'Building Epoch %d/%s dist_mean=%f; min_count=%d(nonzero_count=%d)' % (epoch + 1, config.epoch_build_dict, total_dist_min/total_count, torch.min(total_count_centers), len_count)}, True)
        model._record_bow_dict(centers_val)  # SET_CENTER_VAL
    # CALCULATE TIME
    now = time.time()
    loggers.update_loss({'time': now - T_START, 'time_building_dict': now - T_START}, True)
    info['model'] = model
    print('BOW MAKE DICT DONE')
    iterRunner(info)
