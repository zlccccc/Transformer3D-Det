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
        for epoch in range(config.epoch_build_dict):
            if epoch % config.log_freq == 0:
                loggers.update_loss({'info_out': 'Building Epoch %d/%s ' % (epoch + 1, config.epoch_build_dict)}, True)
            total_sum_centers, total_count_centers = 0, 0
            # t = time.time()
            for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                points = data['point_set'].float().cuda()
                # print('load', time.time() - t)
                # t = time.time()
                with torch.no_grad():  # not useful
                    _, sc_val, cc_val = compute_centers(points, centers_val)
                    total_sum_centers += sc_val
                    total_count_centers += cc_val
                # print(time.time() - t)
            centers_val = total_sum_centers / total_count_centers
        model._record_bow_dict(centers_val)  # SET_CENTER_VAL
    # CALCULATE TIME
    now = time.time()
    loggers.update_loss({'time': now - T_START, 'time_building_dict': now - T_START}, True)
    info['model'] = model
    print('BOW MAKE DICT DONE')
    iterRunner(info)
