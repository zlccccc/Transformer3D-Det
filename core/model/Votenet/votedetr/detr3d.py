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
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)

from transformer3D import build_transformer, MLP
from position_encoding import build_position_encoding
# from .transformer3D import build_transformer


class DETR3D(nn.Module):  # just as a backbone; encoding afterward
    """ This is the DETR module that performs object detection """
    def __init__(self, config_transformer, input_channels, class_output_shape, bbox_output_shape, aux_loss=False):  # new: from config_transformer
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
        transformer_type = config_transformer.get('transformer_type', 'enc_dec')
        self.transformer_type = transformer_type
        if 'dec' in transformer_type:
            num_queries = config_transformer.num_queries
            self.num_queries = num_queries
        # seed_attention: Center; SizeType; Direction; Offset
        self.seed_attention = config_transformer.get('seed_attention', False)
        if self.seed_attention:
            config_transformer.offset_size = bbox_output_shape

        self.transformer = build_transformer(config_transformer)
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Linear(input_channels, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, class_output_shape)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, bbox_output_shape, 3)  # TODO Change MLP LN

        if 'dec' in transformer_type:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        else:
            self.query_embed = None

        self.pos_embd_type = config_transformer.position_embedding
        self.mask_type = config_transformer.get('mask', 'detr_mask')

        self.weighted_input = config_transformer.get('weighted_input', False)
        if self.weighted_input:
            print('[INFO!] Use Weighted Input!')
        if self.seed_attention:
            self.seed_proj = nn.Linear(input_channels, hidden_dim)
            print('[INFO!] Use Seed Attention!')

        if self.pos_embd_type in ['self', 'none']:
            self.pos_embd = None
        else:
            self.pos_embd = build_position_encoding(config_transformer.position_embedding, hidden_dim, config_transformer.input_dim)
        self.aux_loss = aux_loss

    def forward(self, xyz, features, output, seed_xyz=None, seed_features=None, decode_vars=None):  # insert into output
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
        # GET MASK

        if self.mask_type == 'detr_mask':
            mask = torch.zeros(B, N).bool().to(xyz.device)
            src_mask = None
        elif self.mask_type == 'no_mask':
            mask = None
            src_mask = None
        elif self.mask_type.split('_')[0] == 'near':
            near_kth = int(self.mask_type.split('_')[1])
            # print('mask_type: get nearest kth', near_kth, flush=True)
            mask = None
            # mask = torch.zeros(B, N).bool().to(xyz.device)
            src_mask = torch.zeros(B, N, N).to(xyz.device) - 1e9
            A = xyz[:, None, :, :].repeat(1, N, 1, 1)
            B = xyz[:, :, None, :].repeat(1, 1, N, 1)
            # print(A.shape, B.shape, '<< mask A and B shape', flush=True)
            dist = torch.sum((A - B).pow(2), dim=-1)
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=1, largest=False, sorted=False)
            # print(dist_min.shape, dist_pos.shape, ' << dist min shape', flush=True)
            src_mask.scatter_(1, dist_pos, 0)
        else:
            raise NotImplementedError(self.mask_type)
        # print(mask, ' <<< mask')
        seed_embd = None
        if self.pos_embd_type == 'self':
            pos_embd = self.input_proj(features)
        elif self.pos_embd_type == 'none':
            pos_embd = None
        else:
            pos_embd = self.pos_embd(xyz)
            if seed_xyz is not None:
                seed_embd = self.pos_embd(seed_xyz)
        # print(xyz, features, '<< before transformer; features not right')
        features = self.input_proj(features)
        # print(features.shape, features.mean(), features.std(), '<< features std and mean')
        query_embd_weight = self.query_embed.weight if self.query_embed is not None else None

        if self.seed_attention: # TODO doit
            assert seed_xyz is not None
            assert seed_features is not None
            seed_features = self.seed_proj(seed_features)
            value = self.transformer(features, mask, query_embd_weight, pos_embd, src_mask, src_position=xyz, seed_position=seed_xyz, seed_feat=seed_features, seed_embed=seed_embd, decode_vars=decode_vars)
        else:
            assert seed_xyz is None
            assert seed_features is None
            if self.weighted_input:  #TODO doit
                value = self.transformer(features, mask, query_embd_weight, pos_embd, src_mask=src_mask, src_position=xyz)
            else:
                value = self.transformer(features, mask, query_embd_weight, pos_embd, src_mask=src_mask)

        # return: dec_layer * B * Query * C
        if 'dec' in self.transformer_type or self.transformer_type.split(';')[-1] == 'deformable':
            hs = value[0]  # features_output
        elif self.transformer_type in ['enc']:  # TODO THIS IS NOT RIGHT! LAYER TO BE DONE
            hs = value
        else:
            raise NotImplementedError(self.transformer_type)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        # outputs_coord = outputs_coord.sigmoid()
        # print(outputs_class.shape, outputs_coord.shape, 'output coord and class')
        if 'dec' in self.transformer_type or self.transformer_type.split(';')[-1] == 'deformable':
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}  # final
            if self.aux_loss:
                output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

            if self.weighted_input or self.seed_attention: # sum with attention weight (just for output!)
                weighted_xyz = value[2]  # just weighted
                # print(weighted_xyz.shape, '<<< weighted xyz value!', flush=True)  # update: it is right!
                output['transformer_weighted_xyz_all'] = weighted_xyz
                output['transformer_weighted_xyz'] = weighted_xyz[-1]  # just sum it
            else:
                assert len(value) == 2
        else:
            output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}  # final
        return output

    # @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


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
