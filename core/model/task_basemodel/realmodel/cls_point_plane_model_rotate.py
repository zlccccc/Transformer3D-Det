import torch
import torch.nn.functional as F
from core.model.task_basemodel.taskmodel.cls_model import cls_module
from core.model.transform.align_transforms_3D import get_transformlist, apply_transform


class cls_plane_module_rotate(cls_module):
    def __init__(self, config):  # MUST HAVE CONFIG(transform)
        print('Should Have transform in config (init get transform list)')
        super(cls_plane_module_rotate, self).__init__()
        if 'transform' in config.keys():
            self.traintransform, self.testtransfrom = get_transformlist(config.transform)
        else:
            raise NotImplementedError('transform should in model')

    def loss_plane_rotate(self, input, output):
        plane, plane_shift = output['plane'], output['plane_shift']
        plane_should = torch.bmm(plane, input['forwardAffineMatShift'].type_as(plane))
        # print(plane_should.shape, input['forwardAffineMatShift'].type_as(plane).shape, plane_shift.shape, '<-   ----   shape')
        norm_1 = torch.sqrt(torch.sum(plane[:, :, :3] ** 2, dim=2)).reshape(plane.shape[0], plane.shape[1], 1)
        norm_2 = torch.sqrt(torch.sum(plane_should[:, :, :3] ** 2, dim=2)).reshape(plane_should.shape[0], plane_should.shape[1], 1)
        plane = plane / norm_1
        plane_should = plane_should / norm_2
        # print(plane_should.shape, plane_shift.shape, '<-   ----   shape')
        leng = (plane_shift - plane_should)
        distance = torch.min(torch.abs(leng), dim=2).values
        dist_mean = torch.mean(distance, dim=1)
        return dist_mean  # batch*size

    def loss_plane(self, input, output):
        point = input['point_set']
        out = output['plane']
        norm = torch.sqrt(torch.sum(out[:, :, :3] ** 2, dim=2)).reshape(out.shape[0], out.shape[1], 1)
        out = out / norm
        point_mult = torch.cat([point[:, :, :3], torch.ones(point.shape[0], point.shape[1], 1).cuda()], dim=2)
        leng = torch.bmm(point_mult, out.permute(0, 2, 1))
        distance = torch.min(torch.abs(leng), dim=2).values
        dist_mean = torch.mean(distance, dim=1)
        return dist_mean  # batch*size

    def _before_forward(self, input):
        # print('before forward: apply transform')
        if self.mode == 'train':
            out = apply_transform(self.traintransform, input['point_set'])  # transform
        elif self.mode == 'test':
            out = apply_transform(self.testtransfrom, input['point_set'])  # transform
        else:
            raise NotImplementedError('before forward: not supported type', self.mode)
        for name, value in out.items():  # merge out and input(TODO check it)
            input[name] = value
            # print(name, value)
        return input

    def calculate_loss(self, input, output):
        output = super().calculate_loss(input, output)
        dist_mean = self.loss_plane(input, output)
        dist_rotate = self.loss_plane_rotate(input, output)
        output['plane_loss'] = torch.mean(dist_mean) * 0
        output['rotate_loss'] = torch.mean(dist_rotate) * 0
        loss = output['loss']
        loss += output['plane_loss']
        output['loss'] = loss
        return output

    def calculate_error(self, input, output):
        output = super().calculate_error(input, output)
        dist_mean = self.loss_plane(input, output)
        # print(dist_mean)
        output['plane_error'] = torch.sum(dist_mean) * 100
        return output
