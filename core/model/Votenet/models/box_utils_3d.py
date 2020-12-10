
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch


def box_area(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    return (x1 - x0) * (y1 - y0) * (z1 - z0)


def box_c2p(x):
    x_c, y_c, z_c, x_w, y_w, z_w = x.unbind(-1)
    b = [(x_c - 0.5 * x_w), (y_c - 0.5 * y_w), (z_c - 0.5 * z_w),
         (x_c + 0.5 * x_w), (y_c + 0.5 * y_w), (z_c + 0.5 * z_w)]
    return torch.stack(b, dim=-1)


def box_p2c(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
         (x1 - x0), (y1 - y0), (z1 - z0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    assert(boxes1.shape == boxes2.shape)

    lt = torch.max(boxes1[:, :3], boxes2[:, :3])  # [N, 3]
    rb = torch.min(boxes1[:, 3:], boxes2[:, 3:])  # [N, 3]

    wh = (rb - lt).clamp(min=0)  # [N, 3]
    inter = wh[:, 0] * wh[:, 1] * wh[:, 2]  # [N, 3]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, z0, x1, y1, z1] format

    Returns a N matrix (iou)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # print(boxes2.shape, boxes2.shape, '<< giou gen') # TODO
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, 0] * wh[:, 1] * wh[:, 2]

    return iou - (area - union) / area

def iou_loss(gt, pred, type='giou'):
    if type == 'giou':
        return generalized_box_iou(gt, pred)
    elif type == 'iou':
        return box_iou(gt, pred)[0]
    else:
        raise NotImplementedError(type)
