import torch
import torch.nn.functional as F
from core.model.task_basemodel.taskmodel.cls_model import cls_module


class cls_plane_module(cls_module):
    def __init__(self):
        super(cls_plane_module, self).__init__()

    def calculate_loss(self, input, output):
        output = super().calculate_loss(input, output)
        point = input['point_set']
        out = output['plane']
        norm = torch.sqrt(torch.sum(out[:, :, :3] ** 2, dim=2)).reshape(out.shape[0], out.shape[1], 1)
        # print(norm)
        out = out / norm
        # print(point.shape, torch.ones(point.shape[0], point.shape[1], 1).cuda())
        point_mult = torch.cat([point[:, :, :3], torch.ones(point.shape[0], point.shape[1], 1).cuda()], dim=2)
        # print(point_mult.shape, out.shape)
        leng = torch.bmm(point_mult, out.permute(0, 2, 1))
        distance = torch.min(torch.abs(leng), dim=2).values
        # print(distance)
        output['plane_loss'] = torch.mean(distance) * 1000
        loss = output['loss']
        loss += output['plane_loss']
        output['loss'] = loss
        return output

    def calculate_error(self, input, output):
        output = super().calculate_error(input, output)
        point = input['point_set']
        out = output['plane']
        norm = torch.sqrt(torch.sum(out[:, :, :3] ** 2, dim=2)).reshape(out.shape[0], out.shape[1], 1)
        out = out / norm
        point_mult = torch.cat([point[:, :, :3], torch.ones(point.shape[0], point.shape[1], 1).cuda()], dim=2)
        leng = torch.bmm(point_mult, out.permute(0, 2, 1))
        distance = torch.min(torch.abs(leng), dim=2).values
        dist_mean = torch.mean(distance, dim=1)
        # print(dist_mean)
        output['plane_error'] = torch.sum(dist_mean) * 100
        return output
