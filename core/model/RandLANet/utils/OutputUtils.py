import numpy as np
import torch
import time
import os
from core.utils.utils import create_logger
from core.data.dataloader.Sementicdataloader.utils.helper_ply import read_ply
import torch.nn.functional as F


class Semantic3DModelTester:
    def __init__(self, model, dataloader):
        # Add a softmax operation for predictions
        self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
                           for l in dataloader.input_trees['test']]

        self.log_out = create_logger('log/log_test_%s.txt', dataloader.name)  # TODO CHANGE IT(NOT OKAY; DDP)

    def test(self, model, dataloader, num_votes=100):
        # Smoothing parameter for votes
        test_smooth = 0.98

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = os.path.join('test', saving_path.split('/')[-1])
        os.makedirs(test_path) if not os.path.exists(test_path) else None
        os.makedirs(os.path.join(test_path, 'predictions')) if not os.path.exists(os.path.join(test_path, 'predictions')) else None
        os.makedirs(os.path.join(test_path, 'probs')) if not os.path.exists(os.path.join(test_path, 'probs')) else None

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            for it, sample in enumerate(dataloader):
                with torch.no_grad():  # no tracking
                    output = model(sample)

                point_idx = sample['input_inds']
                cloud_idx = sample['cloud_inds']
                stacked_probs = F.softmax(output['logits'], dim=-1).cpu().numpy()  # for every point prediction

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                self.log_out.log('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(
                    epoch_id, step_id, np.min(dataloader.min_possibility['test'])), self.log_out)

            # Save predicted cloud
            new_min = np.min(dataloader.min_possibility['test'])
            self.log_out.log('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

            if last_min + 4 < new_min:

                print('Saving clouds')

                # Update last_min
                last_min = new_min

                # Project predictions
                print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                t1 = time.time()
                files = dataloader.test_files
                i_test = 0
                for i, file_path in enumerate(files):
                    # Get file
                    points = self.load_evaluation_points(file_path)
                    points = points.astype(np.float16)

                    # Reproject probs
                    probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                    proj_index = dataloader.test_proj[i_test]

                    probs = self.test_probs[i_test][proj_index, :]

                    # Insert false columns for ignored labels
                    probs2 = probs
                    for l_ind, label_value in enumerate(dataloader.label_values):
                        if label_value in dataloader.ignored_labels:
                            probs2 = np.insert(probs2, l_ind, 0, axis=1)

                    # Get the predicted labels
                    preds = dataloader.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                    # Save plys
                    cloud_name = file_path.split('/')[-1]

                    # Save ascii preds
                    ascii_name = os.path.join(test_path, 'predictions', dataloader.ascii_files[cloud_name])
                    np.savetxt(ascii_name, preds, fmt='%d')
                    self.log_out.log(ascii_name + 'has saved', self.log_out)
                    i_test += 1

                t2 = time.time()
                # ITER DONE
                print('Done in {:.1f} s\n'.format(t2 - t1))
                return

            epoch_id += 1
            step_id = 0
            continue
        print('Testing: done')

    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
