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

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer3D(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0,
                 activation="gelu", normalize_before=False,
                 return_intermediate_dec=False, have_decoder=True, attention_type='default', deformable_type=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.have_decoder = have_decoder
        if have_decoder:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, attention_type=attention_type, deformable_type=deformable_type)
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

    def forward(self, src, mask, query_embed, pos_embed, static_feat=None, src_mask=None, src_position=None):  #TODO ADD STATIC_FEAT(WEIGHTED SUM or fc)
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

        memory = self.encoder(src, src_key_padding_mask=mask, mask=src_mask, pos=pos_embed)
        # print('encoder done ???')
        if not self.have_decoder:  # TODO LOCAL ATTENTION
            return memory.permute(1, 0, 2)  # just return it
        # to get decode layer
        if self.attention_type.split(';')[-1] == 'deformable':
            assert query_embed is None, 'deformable: query embedding should be None'
            query_embed = torch.zeros_like(src)
            if pos_embed is not None:
                query_embed = pos_embed
                # print(query_embed, '>>query embed', flush=True)
            tgt = src
            tgt_mask = src_mask
            # print(query_embed.shape, '<<< query embedding shape', flush=True)
        else:  # just Add It
            query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
            tgt = torch.zeros_like(query_embed)

        if src_position is not None:
            src_position = src_position.permute(1, 0, 2)
        decoder_output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed, src_position=src_position, tgt_position=src_position)
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
                tgt_position: Optional[Tensor] = None):
        output = tgt

        intermediate, intermediate_pos = [], []

        for layer in self.layers:
            output, nxt_position = layer(output, memory, tgt_mask=tgt_mask,
                                         memory_mask=memory_mask,
                                         tgt_key_padding_mask=tgt_key_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask,
                                         pos=pos, query_pos=query_pos, src_position=src_position, tgt_position=tgt_position)
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
                             'input', 'interpolation_xyz', 'interpolation_xyz_0.1'], 'attn_type should be nearby|interpolation'
        self.attn_type = attn_type
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    
    def forward(self, query, key, value, attn_mask, key_padding_mask, src_position, tgt_position):
        if self.attn_type in ['input']: # just using attn_mask from input
            return attn_with_batch_mask(self.attention, query, key, src=value, src_mask=attn_mask,
                                        src_key_padding_mask=key_padding_mask)
        N, B, C = src_position.shape
        X = src_position[None, :, :, :].repeat(N, 1, 1, 1)
        Y = tgt_position[:, None, :, :].repeat(1, N, 1, 1)
        dist = torch.sum((X - Y).pow(2), dim=-1)
        dist = dist.permute(2, 0, 1)
        # print(dist.shape, '<<< dist.shape', flush=True)
        if self.attn_type in ['nearby']:
            assert attn_mask is None, 'positional attn: mask should be none'
            near_kth = 5
            # TODO GETID and INTERPOLATION
            # print('Using MultiheadPositionalAttention', near_kth, ' <<< near kth', flush=True)
            # print(A.shape, B.shape, '<< mask A and B shape', flush=True)
            # print(dist_min.shape, dist_pos.shape, ' << dist min shape', dist_pos[0, 0:2], flush=True)
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=1, largest=False, sorted=False)
            src_mask = torch.zeros(B, N, N).to(dist.device) - 1e9
            src_mask.scatter_(1, dist_pos, 0)
            ret = attn_with_batch_mask(self.attention, query, key, src=value, src_mask=src_mask,
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
            src_mask = torch.zeros(B, N, N).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, weight.exp())
            ret = attn_with_batch_mask(self.attention, query, key, src=value, src_mask=src_mask,
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
            src_mask = torch.zeros(B, N, N).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, -dist_min)
            ret = attn_with_batch_mask(self.attention, query, key, src=value, src_mask=src_mask,
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
            src_mask = torch.zeros(B, N, N).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, weight.exp())
            attn = attn_with_batch_mask(self.attention, query, key, src=value, src_mask=src_mask,
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
            src_mask = torch.zeros(B, N, N).to(dist.device) - 1e9
            src_mask.scatter_(2, dist_pos, 0)
            ret = attn_with_batch_mask(self.attention, query, key, src=value, src_mask=src_mask,
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
        else:
            raise NotImplementedError(self.attn_type)
        # self.attention(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, attention_type='default', deformable_type=None):
        super().__init__()
        attn_split = attention_type.split(';')
        if len(attn_split) == 1:
            attention_input = 'input'
        else:
            attention_input = attn_split[0]
            assert len(attn_split) == 2, 'len(attention_type) should be 1 or 2'
        attention_type = attn_split[-1]
        self.attention_type = attention_type
        print('transformer: Using Decoder transformer type', attention_input, attention_type)
        if attention_type == 'default':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, attn_type=attention_input)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif attention_type.split(';')[-1] == 'deformable':
            self.linear_offset = nn.Linear(d_model, 3)  # center forward
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
                     tgt_position: Optional[Tensor] = None):
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
            # print(self.linear_offset.weight.data.max(), self.linear_offset.bias.data.max(), ' << linear_offset shape max')
            # print(offset.shape, tgt_position.shape, offset.max(), '<< offset shape', flush=True)
            tgt_position = tgt_position + offset
            attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       src_position=src_position,
                                       tgt_position=tgt_position)
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

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                src_position: Optional[Tensor] = None,
                tgt_position: Optional[Tensor] = None):
        if self.normalize_before:
            raise NotImplementedError('todo: detr - decoder - normalize_before (wrong when normalize_before_with_tgt_position_encoding)')
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, src_position, tgt_position)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, src_position, tgt_position)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    transformer_type = args.get('transformer_type', 'enc_dec')
    print('[build transformer] Using transformer type', transformer_type)
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
            have_decoder=True,  # using input position
            attention_type=transformer_type,
            deformable_type=args.get('deformable_type','nearby')
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
