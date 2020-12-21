# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import math
import copy
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


# copy from proposal codes TODO!
def decode_scores_boxes(output_dict, end_points, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias=False, quality_channel=False):
    pred_boxes = output_dict['pred_boxes']
    batch_size = pred_boxes.shape[0]
    num_proposal = pred_boxes.shape[1]
    bbox_args_shape = 3+num_heading_bin*2+num_size_cluster*4
    if quality_channel:
        bbox_args_shape += 1
    assert pred_boxes.shape[-1] == bbox_args_shape, 'pred_boxes.shape wrong'

    if center_with_bias:
        # print('CENTER ADDING VOTE-XYZ', flush=True)
        base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        # print('Using Center With Bias', output_dict.keys())
        if 'transformer_weighted_xyz' in output_dict.keys():
            end_points['transformer_weighted_xyz_all'] = output_dict['transformer_weighted_xyz_all']  # just for visualization
            transformer_xyz = output_dict['transformer_weighted_xyz']
            # print(transformer_xyz[0, :4], base_xyz[0, :4], 'from vote helper', flush=True)
            # print(center.shape, transformer_xyz.shape)
            transformer_xyz = nn.functional.pad(transformer_xyz, (0, 3+num_heading_bin*2+num_size_cluster*4-transformer_xyz.shape[-1]))
            pred_boxes = pred_boxes + transformer_xyz  # residual
        else:
            base_xyz = nn.functional.pad(base_xyz, (0, num_heading_bin*2+num_size_cluster*4))
            pred_boxes = pred_boxes + base_xyz  # residual
    else:
        raise NotImplementedError('center without bias(for decoder): not Implemented')

    center = pred_boxes[:,:,0:3] # (batch_size, num_proposal, 3) TODO RESIDUAL
    end_points['center'] = center

    heading_scores = pred_boxes[:,:,3:3+num_heading_bin]  # theta; todo change it
    heading_residuals_normalized = pred_boxes[:,:,3+num_heading_bin:3+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = pred_boxes[:,:,3+num_heading_bin*2:3+num_heading_bin*2+num_size_cluster]
    # Bxnum_proposalxnum_size_clusterx3 TODO NEXT WORK REMOVE BBOX-SIZE-DEFINED
    size_residuals_normalized = pred_boxes[:,:,3+num_heading_bin*2+num_size_cluster:3+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3])
    # size_residuals_normalized = size_residuals_normalized.sigmoid() * 2 - 1
    # size_residuals_normalized = size_residuals_normalized.atan() / math.pi  # -0.5 to 0.5
    # print('size normalized value max and min', size_residuals_normalized.max(), size_residuals_normalized.min(), size_residuals_normalized.std(), flush=True)
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    mean_size = torch.from_numpy(mean_size_arr.astype(np.float32)).type_as(pred_boxes).unsqueeze(0).unsqueeze(0)
    end_points['size_residuals'] = size_residuals_normalized * mean_size
    # print(3+num_heading_bin*2+num_size_cluster*4, ' <<< bbox heading and size tensor shape')
    return end_points


class Transformer3D(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0,
                 activation="gelu", normalize_before=False,
                 return_intermediate_dec=False, have_encoder=True, have_decoder=True, attention_type='default', deformable_type=None, offset_size=3):
        super().__init__()

        self.have_encoder = have_encoder
        if have_encoder:
            print('[Attention:] The Transformer Model Have Encoder Module')
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.have_decoder = have_decoder
        if have_decoder:
            print('[Attention:] The Transformer Model Have Decoder Module')
            self.offset_size = offset_size
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, attention_type=attention_type, deformable_type=deformable_type, offset_size=offset_size)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)
            self.attention_type = attention_type

        # self._reset_parameters()  # for fc

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # TODO ADD STATIC_FEAT(WEIGHTED SUM or fc)
    def forward(self, src, mask, query_embed, pos_embed, static_feat=None, src_mask=None, src_position=None, seed_position=None, seed_feat=None, seed_embed=None, decode_vars=None):
        # flatten BxNxC to NxBxC
        # print(src.shape, pos_embed.shape, query_embed.shape, '<<< initial src and query shape', mask.shape, flush=True)
        B, N, C = src.shape
        src = src.permute(1, 0, 2)
        if pos_embed is not None:
            pos_embed = pos_embed.permute(1, 0, 2)
        # print(mask.shape, '<< mask shape, from transformer3d.py', src_mask, flush=True)
        # print('<< mask shape, from transformer3d.py', src_mask, flush=True)
        # mask = None
        # print(mask)
        # print(src.shape, pos_embed.shape, query_embed.shape, mask.shape, '<<< src and post shape')
        # print(src, pos_embed[0], '<<< src and post shape')
        # print(src)
        # print(src.mean(), src.std(), '<< transformer input std value features mean and std', flush=True)
        if self.have_encoder:
            assert seed_position is None
            assert seed_feat is None
            memory = self.encoder(src, src_key_padding_mask=mask, mask=src_mask, pos=pos_embed)
        else:
            if seed_feat is None:
                memory = src
            else:
                memory = seed_feat.permute(1, 0, 2)
        # print('encoder done ???')
        if not self.have_decoder:  # TODO LOCAL ATTENTION
            return memory.permute(1, 0, 2)  # just return it

        # to get decode layer TODO
        if self.attention_type.split(';')[-1] == 'deformable':
            assert query_embed is None, 'deformable: query embedding should be None'
            query_embed = torch.zeros_like(src)
            # if pos_embed is not None:
            #     query_embed = pos_embed
            #     print(query_embed, '>>query embed', flush=True)
            tgt = src
            tgt_mask = src_mask
            # print(query_embed.shape, '<<< query embedding shape', flush=True)
        else:  # just Add It
            query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
            tgt = torch.zeros_like(query_embed)

        if src_position is not None:
            src_position = src_position.permute(1, 0, 2)
        if seed_position is not None:
            seed_position = seed_position.permute(1, 0, 2)
            tgt_position = nn.functional.pad(src_position, (0, self.offset_size-3))
            pos_embed = seed_embed
            if pos_embed is not None:
                pos_embed = pos_embed.permute(1, 0, 2)
            # if seed_embed is not None:
            #     print(seed_embed.shape, '<< seed embed')
            # print('pad seedpos', self.offset_size-3)
        else:
            tgt_position = src_position
            seed_position = src_position

        # print(decode_vars, '<< decode vars FOR TEST  TODO')
        # self-attention:  tgt -> tgt
        # cross-attention: src/memory -> tgt
        decoder_output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed, src_position=seed_position, tgt_position=tgt_position, decode_vars=decode_vars)
        # print(hs.transpose(1,2).shape, memory.shape, '<< final encoder and decode shape', flush=True)

        if src_position is not None:
            hs, finpos = decoder_output
            # print(hs.shape, memory.shape, finpos.shape, '<<< fin pos shape', flush=True)
            # print((finpos[-1] - src_position).max(), '  <<<  finpos shift', flush=True)
            return hs.transpose(1, 2), memory.permute(1, 0, 2), finpos.transpose(1, 2) # .view(B, N, C)
        else:
            hs = decoder_output
        return hs.transpose(1, 2), memory.permute(1, 0, 2)  # .view(B, N, C)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            # print(output, '<< ENCODER output layer??')

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                src_position: Optional[Tensor] = None,
                tgt_position: Optional[Tensor] = None,
                decode_vars: Optional = None):
        output = tgt

        intermediate, intermediate_pos = [], []

        for layer in self.layers:
            output, nxt_position = layer(output, memory, tgt_mask=tgt_mask,
                                         memory_mask=memory_mask,
                                         tgt_key_padding_mask=tgt_key_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask,
                                         pos=pos, query_pos=query_pos, src_position=src_position, tgt_position=tgt_position, decode_vars=decode_vars)
            # print((tgt_position-nxt_position).abs().max(), '<< xyz, bias, from transformer')
            # print(output.shape, '<< output shape', tgt_position.shape, '<< tgt shape', flush=True)
            tgt_position = nxt_position
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_pos.append(tgt_position)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_pos)

        return output.unsqueeze(0), tgt_position.unsqueeze(0)


def attn_with_batch_mask(layer_attn, q, k, src, src_mask, src_key_padding_mask):
    bs, src_arr, attn_arr = q.shape[1], [], []
    for i in range(bs):
        key_mask, attn_mask = None, None
        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask[i:i+1]
        if src_mask is not None:
            attn_mask = src_mask[i]
        batch_attn = layer_attn(q[:, i:i+1, :], k[:, i:i+1, :], value=src[:, i:i+1, :], attn_mask=attn_mask,
                                key_padding_mask=key_mask)
        # print(batch_attn[1].sum(dim=-1))  # TODO it is okay to make a weighted sum
        # print(batch_attn[1], attn_mask, flush=True
        src_arr.append(batch_attn[0])
        attn_arr.append(batch_attn[1])
    src2 = torch.cat(src_arr, dim=1)
    attn = torch.cat(attn_arr, dim=0)
    return src2, attn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # print(q.shape, src.shape, src_mask.shape, '<< forward post shape; todo', flush=True)
        src2 = attn_with_batch_mask(self.self_attn, q, k, src=src, src_mask=src_mask,
                                    src_key_padding_mask=src_key_padding_mask)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # print(q, k, '<< forward!!')
        # print(src2, '<< forward')
        # exit()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class MultiheadPositionalAttention(nn.Module):  # nearby points
    def __init__(self, d_model, nhead, dropout, attn_type='nearby'):  # nearby; interpolation
        super().__init__()
        assert attn_type in ['nearby', 'interpolation', 'interpolation_10', 'near_interpolation', 'dist', 'dist_10',
                             'input', 'interpolation_xyz', 'interpolation_xyz_0.1',
                             'inside_bbox', 'inside_around_bbox', 'inside_around_bbox_max', 'inside_direction_bbox'], 'attn_type should be nearby|interpolation'
        self.attn_type = attn_type
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


    @staticmethod
    def rotz_batch_pytorch(t):
        """
        Rotation about the z-axis
        :param t: (x1,x2,...,xn)
        :return: output:(x1,x2,...,xn,3,3)
        """
        input_shape = t.shape  # (B, num_proposal)
        output = torch.zeros(tuple(list(input_shape)+[3,3])).type_as(t)
        c = torch.cos(t)
        s = torch.sin(t)
        # Attention ~ 这里的rot_mat是已经转置过的matrix，是为了进行 x'A' = (Ax)'
        # [[cos(t), -sin(t), 0],
        #  [sin(t), cos(t),   0],
        #  [0,     0,        1]]
        output[...,0,0] = c
        output[...,0,1] = -s
        output[...,1,0] = s
        output[...,1,1] = c
        output[...,2,2] = 1
        return output


    def forward(self, query, key, value, attn_mask, key_padding_mask, src_position, tgt_position, decode_vars=None):  # TODO Check Decode Vars
        if self.attn_type in ['input']: # just using attn_mask from input
            return attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=attn_mask,
                                        src_key_padding_mask=key_padding_mask)
        # print(query.shape, key.shape, value.shape, '<< cross attn shape', flush=True)
        N, B, C = src_position.shape
        N2, B2, C2 = tgt_position.shape
        if C == 3 and C2 != 3:
#def decode_scores_boxes(output_dict, end_points, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias=False, quality_channel=False):
            # import ipdb
            # ipdb.set_trace()
            # self: transformer_xyz; so wo should add zero (transformer_weighted_xyz; aggregated_vote_xyz)
            tgt_position = tgt_position.permute(1, 0, 2)
            tgt_fin_dict = decode_scores_boxes({'pred_boxes': torch.zeros_like(tgt_position)},
                                               {'aggregated_vote_xyz': decode_vars['aggregated_vote_xyz'], 'transformer_weighted_xyz': tgt_position.detach()},
                                               decode_vars['num_heading_bin'], decode_vars['num_size_cluster'], decode_vars['mean_size_arr'],
                                               center_with_bias=True)
            # print(tgt_fin_dict, '<< fin tgt list; for obj-mask use')

        if self.attn_type in ['inside_bbox', 'inside_around_bbox', 'inside_around_bbox_max', 'inside_direction_bbox']:
            tgt_position = tgt_fin_dict['center'].permute(1, 0, 2)
            Y = src_position[:, None, :, :].repeat(1, N2, 1, 1)
            X = tgt_position[None, :, :, :].repeat(N, 1, 1, 1)
            shift = Y-X  # Center --to-> Src
            # 'heading_scores', 'heading_residuals'  ==> size
            pred_heading_scores, pred_heading_residuals = tgt_fin_dict['heading_scores'], tgt_fin_dict['heading_residuals']
            pred_heading_ind = torch.argmax(pred_heading_scores, -1).detach()  # (B, num_proposal)
            pred_heading_class = pred_heading_ind.float()*(2*np.pi/float(decode_vars['num_heading_bin']))
            pred_heading_residuals = torch.gather(pred_heading_residuals, 2, pred_heading_ind[:, :, None]).squeeze(-1).detach()  # (B, num_proposal)
            pred_heading = pred_heading_class + pred_heading_residuals  # (B, num_proposal)
            # 'size_scores', 'size_residuals'  ==> size
            pred_size_scores, pred_size_residuals = tgt_fin_dict['size_scores'], tgt_fin_dict['size_residuals']
            mean_size_arr = decode_vars['mean_size_arr']
            mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).type_as(pred_size_scores).unsqueeze(0).repeat(B,1,1)  # B, (num_size_cluster, 3)
            pred_size_ind = torch.argmax(pred_size_scores, -1).detach()  # (B, num_proposal)
            pred_size_class = torch.gather(mean_size_arr_expanded, 1, pred_size_ind[:, :, None].repeat(1, 1, 3))  # (B, num_proposal, 3)
            pred_size_residuals = torch.gather(pred_size_residuals, 2, pred_size_ind[:, :, None, None].repeat(1, 1, 1, 3)).squeeze(2)  # (B, num_proposal, 3)
            pred_size = pred_size_class + pred_size_residuals  # (B, num_proposal)
            # shift back
            rotz = self.rotz_batch_pytorch(pred_heading).reshape(B*N2, 3, 3)
            shift = shift.permute(2, 1, 0, 3)  # N*N2*B*3 => B*N2*N*3
            # print(shift.shape, '<< shift shape 0')
            shift = shift.reshape(B*N2, N, 3)  # B*N2*N*3 => (B*N2)*N*3
            # print(shift.shape, '<< shift shape 1')
            shift = torch.bmm(shift, rotz)
            # print(shift.shape, '<< shift shape 2')
            shift = shift.view(B, N2, N, 3)
            # print(pred_size.shape, shift.shape, '<< TODO COMPARE')
            src_mask = torch.zeros(shift.shape[:-1]).to(shift.device) - 1e9
            scaled_shift = torch.abs(shift) / pred_size[:, :, None, :] * 2
            scaled_shift_abs = torch.abs(shift)
            mask = scaled_shift_abs <= 1.5  # 1.5 times bigger
            mask = mask[:, :, :, 0] & mask[:, :, :, 1] & mask[:, :, :, 2]
            if self.attn_type == 'inside_bbox':
                src_mask[mask] = 0
            elif self.attn_type == 'inside_around_bbox':
                src_mask[mask] = -torch.sum(torch.abs(scaled_shift_abs-1), dim=-1)[mask]
            elif self.attn_type == 'inside_around_bbox_max':
                src_mask[mask] = -torch.min(torch.abs(scaled_shift_abs-1), dim=-1)[0][mask]
            elif self.attn_type == 'inside_direction_bbox':
                assert self.nhead == 8
                mask_0_l, mask_1_l, mask_2_l = src_mask, src_mask, src_mask
                mask_0_r, mask_1_r, mask_2_r = src_mask, src_mask, src_mask
                mask_0_l[mask] = -torch.abs(scaled_shift[:, :, :, 0][mask] + 1)
                mask_1_l[mask] = -torch.abs(scaled_shift[:, :, :, 1][mask] + 1)
                mask_2_l[mask] = -torch.abs(scaled_shift[:, :, :, 2][mask] + 1)
                mask_0_r[mask] = -torch.abs(scaled_shift[:, :, :, 0][mask] - 1)
                mask_1_r[mask] = -torch.abs(scaled_shift[:, :, :, 1][mask] - 1)
                mask_2_r[mask] = -torch.abs(scaled_shift[:, :, :, 2][mask] - 1)
                mask_all, mask_none = src_mask, src_mask
                mask_all[mask] = -torch.min(torch.abs(scaled_shift_abs-1), dim=-1)[0][mask]
                mask_none[mask] = 0
                src_mask_nhead = torch.stack((mask_0_l, mask_0_r, mask_1_l, mask_1_r, mask_2_l, mask_2_r, mask_all, mask_none), dim=1)
                print(src_mask_nhead.shape, '<< attn shape')
                src_mask_nhead = src_mask_nhead.reshape(B * self.nhead, N2, N)
                # Attention:src_mask_nhead shape (B*nhead, N2, N) not okay for some attention-type
                ret = self.attention(query=query, key=key, value=value, attn_mask=src_mask_nhead, key_padding_mask=key_padding_mask)
                print(len(ret), '<< ret', flush=True)
                return ret

            ret = attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=src_mask,
                                        src_key_padding_mask=key_padding_mask)

            # ret = layer_attn(q[:, i:i+1, :], k[:, i:i+1, :], value=src[:, i:i+1, :], attn_mask=mask,
            #                  key_padding_mask=src_key_padding_mask)

            return ret

        if C != 3 and C2 != 3 and C == C2:
            C2 = C = 3  # only xyz is useful
            src_position = src_position[:, :, :3]
            tgt_position = tgt_position[:, :, :3]
        # Using Just XYZ
        assert B2 == B and C2 == C
        Y = src_position[:, None, :, :].repeat(1, N2, 1, 1)
        X = tgt_position[None, :, :, :].repeat(N, 1, 1, 1)
        dist = torch.sum((X - Y).pow(2), dim=-1)
        dist = dist.permute(2, 0, 1)
        # print(dist.shape, '<<< dist.shape', query.shape, key.shape, value.shape, flush=True)
        # import ipdb
        # ipdb.set_trace()
        if self.attn_type in ['nearby']:
            assert attn_mask is None, 'positional attn: mask should be none'
            near_kth = 5
            # TODO GETID and INTERPOLATION
            # print('Using MultiheadPositionalAttention', near_kth, ' <<< near kth', flush=True)
            # print(A.shape, B.shape, '<< mask A and B shape', flush=True)
            # print(dist_min.shape, dist_pos.shape, ' << dist min shape', dist_pos[0, 0:2], flush=True)
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=1, largest=False, sorted=False)
            src_mask = torch.zeros(dist.shape).to(dist.device) - 1e9
            src_mask.scatter_(1, dist_pos, 0)
            src_mask = src_mask.permute(0, 2, 1) # V*k'*q
            ret = attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=src_mask,
                                        src_key_padding_mask=key_padding_mask)
            return ret
        elif self.attn_type in ['interpolation', 'interpolation_10']:  # similiar as pointnet
            assert attn_mask is None, 'positional attn: mask should be none'
            # dist_recip = 1 / (dist + 1e-8)
            # norm = torch.sum(dist_recip, dim=1, keepdim=True)
            # weight = dist_recip / norm
            # print(norm.shape, weight.shape)
            near_kth = 5
            kth_split = self.attn_type.split('_')
            if len(kth_split) == 2:
                near_kth = int(kth_split[-1])
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=-1, largest=False, sorted=False)
            # weight
            dist_recip = 1 / (dist_min + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # B * N * near_kth
            # src_mask
            src_mask = torch.zeros(dist.shape).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, weight.exp())
            src_mask = src_mask.permute(0, 2, 1) # V*k'*q
            ret = attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=src_mask,
                                       src_key_padding_mask=key_padding_mask)
            return ret
        elif self.attn_type in ['dist', 'dist_10']:  # similiar as pointnet
            assert attn_mask is None, 'positional attn: mask should be none'
            # dist_recip = 1 / (dist + 1e-8)
            # norm = torch.sum(dist_recip, dim=1, keepdim=True)
            # weight = dist_recip / norm
            # print(norm.shape, weight.shape)
            near_kth = 5
            kth_split = self.attn_type.split('_')
            if len(kth_split) == 2:
                near_kth = int(kth_split[-1])
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=-1, largest=False, sorted=False)
            # weight
            src_mask = torch.zeros(dist.shape).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, -dist_min)
            src_mask = src_mask.permute(0, 2, 1) # V*k'*q
            ret = attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=src_mask,
                                       src_key_padding_mask=key_padding_mask)
            return ret
        elif self.attn_type in ['interpolation_xyz', 'interpolation_xyz_0.1']:  # similiar as pointnet
            assert attn_mask is None, 'positional attn: mask should be none'
            near_kth = 5
            scale = 0.5
            scale_split = self.attn_type.split('_')
            if len(scale_split) == 3:
                near_scale = float(scale_split[-1])
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=-1, largest=False, sorted=False)
            # weight
            dist_recip = 1 / (dist_min + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # B * N * near_kth
            # src_mask
            src_mask = torch.zeros(dist.shape).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, weight.exp())
            src_mask = src_mask.permute(0, 2, 1) # V*k'*q
            attn = attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=src_mask,
                                       src_key_padding_mask=key_padding_mask)
            attn_map = attn[-1]
            # print(attn_map.shape, query.shape, key.shape, value.shape, src_position.shape, '<< shape!!', flush=True)
            src_xyz_attn = torch.bmm(src_position.permute(1, 2, 0), attn_map)
            src_xyz_attn = src_xyz_attn.permute(2, 0, 1)
            tgt_position = tgt_position * (1 - scale) + src_xyz_attn * scale
            return attn[0], attn[1], tgt_position
        elif self.attn_type in ['near_interpolation']:  # similiar as pointnet
            assert attn_mask is None, 'positional attn: mask should be none'
            near_kth = 5
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=-1, largest=False, sorted=False)
            # src_mask
            src_mask = torch.zeros(dist.shape).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, 0)
            src_mask = src_mask.permute(0, 2, 1) # V*k'*q
            ret = attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=src_mask,
                                       src_key_padding_mask=key_padding_mask)
            # weight
            dist_recip = 1 / (dist_min + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # B * N * near_kth
            weight = weight.permute(1, 2, 0).view(N * near_kth, B)
            dist_pos = dist_pos.permute(1, 2, 0).view(N * near_kth, B)
            dist_repos = torch.gather(value, 0, dist_pos.unsqueeze(-1).repeat(1, 1, value.shape[-1]))
            more = dist_repos.mul(weight.unsqueeze(-1).repeat(1, 1, value.shape[-1]))
            # print(weight, flush=True) # TODO
            more = more.view(N, near_kth, B, -1)
            more = torch.sum(more, dim=1)
            ret[0] = ret[0] * 0.8 + more * 0.2
            return ret
        elif self.attn_type in ['seed_bboxes']:
            raise NotImplementedError(self.attn_type)
            ret = None
            return ret
        else:
            raise NotImplementedError(self.attn_type)
        # self.attention(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, attention_type='default', deformable_type=None, offset_size=3):
        super().__init__()
        attn_split = attention_type.split(';')
        if len(attn_split) == 1:
            attention_input = 'input'
        else:
            attention_input = attn_split[0]
            assert len(attn_split) == 2 or len(attn_split) == 3, 'len(attention_type) should be 1 or 2'
        attention_type = attn_split[-1]
        self.attention_type = attention_type
        print('transformer: Using Decoder transformer type', attention_input, attention_type)
        if attention_type == 'default':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, attn_type=attention_input)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif attn_split[-1] == 'deformable':
            if offset_size != 3:
                self.linear_offset = MLP(d_model, d_model, offset_size, 3, norm=nn.LayerNorm)
            else:
                self.linear_offset = nn.Linear(d_model, offset_size)  # center forward
                self.linear_offset.weight.data.zero_()
                self.linear_offset.bias.data.zero_()
            # print(self.linear_offset.weight.data.max(), '<< linear OFFSET WIEGHT  !')
            assert deformable_type is not None
            src_attn_type = deformable_type
            self.self_attn = MultiheadPositionalAttention(d_model, nhead, dropout=dropout, attn_type=attention_input)
            self.multihead_attn = MultiheadPositionalAttention(d_model, nhead, dropout=dropout, attn_type=src_attn_type)
        else:
            raise NotImplementedError(attention_type)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     src_position: Optional[Tensor] = None,
                     tgt_position: Optional[Tensor] = None,
                     decode_vars: Optional = None):
        if self.attention_type == 'default':
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        elif self.attention_type.split(';')[-1] == 'deformable':
            q = k = self.with_pos_embed(tgt, query_pos)
            attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask,
                                  src_position=tgt_position,
                                  tgt_position=tgt_position)
            tgt2 = attn[0]
            if len(attn) == 3:
                # print('attn from output! TODO')
                tgt_position = attn[2]
            else:
                assert len(attn) == 2, 'attn len should not be 2 or 3'
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # TODO src_position_attention checking
            offset = self.linear_offset(tgt)
            # print(offset.shape, ' <<< offset')
            # print(offset[:5, 1, :6], '<< offset', flush=True)
            # print(self.linear_offset.weight.data.max(), self.linear_offset.bias.data.max(), ' << linear_offset shape max')
            # print(offset.shape, tgt_position.shape, offset.max(), '<< offset shape', flush=True)
            tgt_position = tgt_position + offset
            attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       src_position=src_position,
                                       tgt_position=tgt_position,
                                       decode_vars=decode_vars)
            tgt2 = attn[0]
            # print(tgt2, '<< tgt2')
            if len(attn) == 3:
                # print('attn from input! TODO')
                tgt_position = attn[2]
            else:
                assert len(attn) == 2, 'attn len should not be 2 or 3'
        else:
            raise NotImplementedError(self.attention_type)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, tgt_position


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                src_position: Optional[Tensor] = None,
                tgt_position: Optional[Tensor] = None,
                decode_vars: Optional = None):
        if self.normalize_before:
            raise NotImplementedError('todo: detr - decoder - normalize_before (wrong when normalize_before_with_tgt_position_encoding)')
            # return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
            #                         tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, src_position, tgt_position)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, src_position, tgt_position, decode_vars)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    transformer_type = args.get('transformer_type', 'enc_dec')
    print('[build transformer] Using transformer type', transformer_type)
    print(args, '<< transformer config')
    if transformer_type == 'enc_dec':
        return Transformer3D(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif transformer_type == 'enc':
        return Transformer3D(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=False,
            have_decoder=False,
        )
    elif transformer_type.split(';')[-1] == 'deformable':
        return Transformer3D(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            have_encoder=False,
            have_decoder=True,  # using input position
            attention_type=transformer_type,
            deformable_type=args.get('deformable_type','nearby'),
            offset_size=args.get('offset_size', 3)
        )
    else:
        raise NotImplementedError(transformer_type)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    print(activation, '<< transformer activation', flush=True)  # TODO REMOVE IT
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
        return gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if norm is not None:
            print('Using Norm << MLP', flush=True)
            self.norm = nn.ModuleList(norm(hidden_dim) for i in range(num_layers-1))
        else:
            self.norm = None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.norm is not None:
                x = F.relu(self.norm[i](layer(x))) if i < self.num_layers - 1 else layer(x)
            else:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

