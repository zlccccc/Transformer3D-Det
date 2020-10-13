# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight_dict: dict):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_weight_dict = cost_weight_dict
        for key, value in cost_weight_dict.items():
            assert isinstance(value, (int, float)), 'cost should be a float value'
            assert value >= 0, 'cost weight should be a positive value'

    @torch.no_grad()
    def forward(self, cost_dict: dict):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # 3d iou is hard to calculate (convex hull)
        # i think minimize(loss) just is also okay

        C = None  # Final cost matrix
        for key, batch_value in cost_dict.items():
            assert key in self.cost_weight_dict.keys(), 'cost_dict should have weight'
            real_value = [value * self.cost_weight_dict[key] for value in batch_value]
            if C is None:
                C = real_value
            else:
                for i, val in enumerate(real_value):
                    C[i] = C[i] + val
        # sizes = [len(v["boxes"]) for v in targets]  # size of gt boxes
        indices = [linear_sum_assignment(c.cpu()) for i, c in enumerate(C)]
        # print(indices)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C


def build_matcher(args):
    return HungarianMatcher(args)
