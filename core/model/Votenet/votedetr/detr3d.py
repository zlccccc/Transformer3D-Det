# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

# from util import box_ops
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# from .backbone import build_backbone
# from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
from transformer3D import build_transformer
from position_encoding import build_position_encoding
# from .transformer3D import build_transformer


class DETR3D(nn.Module):  # just as a backbone; encoding afterward
    """ This is the DETR module that performs object detection """
    def __init__(self, config_transformer, input_channels, class_output_shape, bbox_output_shape, aux_loss=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            input_channels: input channel of point cloud features
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        num_queries = config_transformer.num_queries
        self.num_queries = num_queries
        self.transformer = build_transformer(config_transformer)
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, class_output_shape)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, bbox_output_shape, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_embd_type = config_transformer.position_embedding
        self.mask_type = config_transformer.get('mask', 'detr_mask')
        if self.pos_embd_type == 'self':
            self.pos_embd = None
        else:
            self.pos_embd = build_position_encoding(config_transformer.position_embedding, hidden_dim, config_transformer.input_dim)
        self.aux_loss = aux_loss

    def forward(self, xyz, features, output):  # insert into output
        """ The forward expects a Dict, which consists of:
               - input.xyz: [batch_size x N x K]
               - input.features: [batch_size x N x C]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        B, N, _ = xyz.shape
        _, _, C = features.shape
        # maybe detr_mask is equal to None mask
        # import ipdb; ipdb.set_trace()
        if self.mask_type == 'detr_mask':
            mask = torch.zeros(B, N).bool().to(xyz.device)
        elif self.mask_type == 'no_mask':
            mask = None
        else:
            raise NotImplementedError
        # print(mask, ' <<< mask')
        if self.pos_embd_type == 'self':
            pos_embd = self.input_proj(features)
        else:
            pos_embd = self.pos_embd(xyz)
        # print(xyz, features, '<< before transformer; features not right')
        features = self.input_proj(features)
        # print(features.shape, features.mean(), features.std(), '<< features std and mean')
        hs = self.transformer(features, mask, self.query_embed.weight, pos_embd)[0]
        
        # print(hs,'<<after transformer', flush=True) # TODO CHECK IT
        # return: dec_layer * B * Query * C
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        # outputs_coord = outputs_coord.sigmoid()
        # print(outputs_class.shape, outputs_coord.shape, 'output coord and class')
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}  # final
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == "__main__":
    from easydict import EasyDict
    # def __init__(self, config_transformer, input_channels, num_classes, num_queries, aux_loss=False):
    config_transformer = {
        'enc_layers': 6,
        'dec_layers': 6,
        'dim_feedforward': 2048,
        'hidden_dim': 288,
        'dropout': 0.1,
        'nheads': 8,
        'num_queries': 100,
        'pre_norm': False,
        'position_embedding': 'sine'
    }
    config_transformer = EasyDict(config_transformer)
    model = DETR3D(config_transformer, 128, 10, 20)
    xyz = torch.randn(4, 100, 3)
    features = torch.randn(4, 100, 128)
    # xyz = torch.randn(4, 3, 100)
    # features = torch.randn(4, 128, 100)
    out = model(xyz, features, {})
    # print(out)
