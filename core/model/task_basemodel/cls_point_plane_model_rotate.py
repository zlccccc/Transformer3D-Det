import torch
import torch.nn.functional as F
from core.model.task_basemodel.cls_model import cls_module
from core.transforms.align_transforms import ToTensor, RandomAffine, SamplePoints, applyAffineMatrices
from torchvision import transforms


class cls_plane_module_rotate(cls_module):
    def __init__(self):
        super(cls_plane_module_rotate, self).__init__()

    def calculate_loss(self, input, output):
        output = super().calculate_loss(input, output)
        point = input['point_set']
        out = output['plane']
        norm = torch.sqrt(torch.sum(out[:, :, :3] ** 2, dim=2)).reshape(out.shape[0], out.shape[1], 1)
        # print(norm)
        out = out / norm
        # print(point.shape, torch.ones(point.shape[0], point.shape[1], 1).cuda())
        point_mult = torch.cat([point[:, :, :3], torch.ones(point.shape[0], point.shape[1], 1).cuda()], dim=2)
        leng = torch.bmm(point_mult, out.permute(0, 2, 1))
        distance = torch.min(torch.abs(leng), dim=2).values
        # print(distance)
        output['plane_loss'] = torch.mean(distance) * 0
        loss = output['loss']
        loss += output['plane_loss']
        output['loss'] = loss
        return output

    def _get_transform_lists(self, config):
        traintransformslist = [SamplePoints(sample_points=config.sample_points)]
        traintransformslist.append(ToTensor())
        traintransformslist.append(RandomAffine(normalize=config.normalize,
                                                degrees=config.rotate,
                                                translate=config.translate,
                                                scale=config.scale,
                                                jitter=config.jitter,
                                                shear=config.shear,
                                                mirror=config.mirror))
        self.traintransforms = transforms.Compose(traintransformslist)

        testtransformslist = [SamplePoints(sample_points=config.sample_points)]
        testtransformslist.append(ToTensor())
        testtransformslist.append(RandomAffine(normalize=config.normalize,
                                               degrees=None, translate=None,
                                               scale=((config.scale[0] + config.scale[1]) / 2,
                                                      (config.scale[0] + config.scale[1]) / 2),
                                               jitter=None, shear=None,
                                               mirror=False))
        self.testtransforms = transforms.Compose(testtransformslist)

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
