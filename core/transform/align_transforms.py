import numbers
import torchvision.transforms.functional as TF
import numpy as np
import random
import torch


def applyAffineMatrix3D(points: torch.tensor, AffineMatrix: torch.tensor):
    """
    :param points: initial points
    :param AffineMatrix: affine matrix
    :return: result points
    """
    # AffineMatrix: 4*4(only 4*3 useful)
    points = points.cuda()
    cat_tensor = torch.ones(points.shape[0], 1).double().cuda()
    points = torch.cat((points, cat_tensor), dim=1)
    points = torch.mm(points, AffineMatrix.cuda())[:, :3]
    points = points.cpu()
    return points


def applyAffineMatrices(points: torch.tensor, AffineMatrix: torch.tensor):  # for batch
    for _ in range(points.shape[0]):
        shape = points[_].shape
        points[_] = applyAffineMatrix3D(points[_].reshape(-1, 3).double(), AffineMatrix[_]).reshape(shape)
    return points


def getAffineMatrixNormalize(points, type):
    """
    :param points: n*3, get mean position and std(?)
    :param type: farthest
    :return: affine matrix
    """
    forward, backward = torch.eye(4).double(), torch.eye(4).double()
    # print('points,shape', points.shape)
    mean = torch.mean(points, dim=0).double()
    if type == 'farthest':
        dist = torch.sqrt(torch.max((points - mean) ** 2))
        # print(dist)
    elif type == 'mean':
        dist = (points - mean) ** 2
        # print('dist max mean sum mean:', torch.max(dist), torch.min(dist), torch.sum(dist), torch.mean(dist))
        dist = torch.sqrt(torch.mean(dist)) * 2.5
    else:
        raise NotImplementedError('Normalization Error', type)
    if dist:
        forward[0][0] = forward[1][1] = forward[2][2] = 1 / dist
        forward[3][0:3] = -mean / dist
        backward[0][0] = backward[1][1] = backward[2][2] = dist
        backward[3][0:3] = mean
    # print('mean', mean.shape, mean)
    # print('forward', forward)
    # print('backward',backward)
    # print('multiply', torch.mm(forward, backward))
    return forward, backward


def getAffineMatrixScale(scale_ranges):
    """
    :param scale_ranges: (range_low, range_high)
    :return: affine matrix
    """
    if scale_ranges[0] == scale_ranges[1]:
        scale = scale_ranges[0]
    else:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    forward, backward = torch.eye(4).double(), torch.eye(4).double()
    forward *= scale
    backward /= scale
    return forward, backward


def getAffineMatrixShear(shear_range):
    """
    :param shear_range: (range_low, range_high)
    :return: affine matrix
    """
    forward, backward = torch.eye(4).double(), torch.eye(4).double()
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            now_shear = random.uniform(-shear_range, shear_range)
            forward[i][j] = now_shear
    backward = forward.inverse()
    return forward, backward


def getAffineMatrixTranslate(translate_range):
    """
    :param translate_range: (range_low, range_high)
    :return: affine matrix
    """
    forward, backward = torch.eye(4).double(), torch.eye(4).double()
    for i in range(3):
        now_translate = random.uniform(-translate_range, translate_range)
        forward[3][i] = now_translate
        backward[3][i] = -now_translate
    return forward, backward


def getAffineMatrixRotate(rotate_range):
    """
    :param rotate_range:  (range_x, range_y, range_z)
    :return: affine matrix
    """
    range_x, range_y, range_z = rotate_range
    forward, backward = torch.eye(4).double(), torch.eye(4).double()

    tmp_forward, tmp_backward = torch.eye(4).double(), torch.eye(4).double()
    angle_x = (torch.clamp(range_x / 4 * torch.randn(1), -range_x, range_x) / 180 * np.pi).double()
    tmp_forward[0:2, 0:2] = torch.DoubleTensor([[torch.cos(angle_x), -torch.sin(angle_x)],
                                                [torch.sin(angle_x), torch.cos(angle_x)]])
    tmp_backward[0:2, 0:2] = torch.DoubleTensor([[torch.cos(-angle_x), -torch.sin(-angle_x)],
                                                 [torch.sin(-angle_x), torch.cos(-angle_x)]])
    forward = torch.mm(forward, tmp_forward)
    backward = torch.mm(tmp_backward, backward)

    tmp_forward, tmp_backward = torch.eye(4).double(), torch.eye(4).double()
    angle_y = (torch.clamp(range_y / 4 * torch.randn(1), -range_y, range_y) / 180 * np.pi).double()
    tmp_forward[1:3, 1:3] = torch.DoubleTensor([[torch.cos(angle_y), -torch.sin(angle_y)],
                                                [torch.sin(angle_y), torch.cos(angle_y)]])
    tmp_backward[1:3, 1:3] = torch.DoubleTensor([[torch.cos(-angle_y), -torch.sin(-angle_y)],
                                                 [torch.sin(-angle_y), torch.cos(-angle_y)]])
    forward = torch.mm(forward, tmp_forward)
    backward = torch.mm(tmp_backward, backward)

    tmp_forward, tmp_backward = torch.eye(4).double(), torch.eye(4).double()
    angle_z = (torch.clamp(range_z / 4 * torch.randn(1), -range_z, range_z) / 180 * np.pi).double()
    tmp_forward[[0, 0, 2, 2], [0, 2, 0, 2]] = torch.DoubleTensor([torch.cos(angle_z), -torch.sin(angle_z),
                                                                  torch.sin(angle_z), torch.cos(angle_z)])
    tmp_backward[[0, 0, 2, 2], [0, 2, 0, 2]] = torch.DoubleTensor([torch.cos(-angle_z), -torch.sin(-angle_z),
                                                                   torch.sin(-angle_z), torch.cos(-angle_z)])
    forward = torch.mm(forward, tmp_forward)
    backward = torch.mm(tmp_backward, backward)
    # print(torch.mm(tmp_forward, tmp_backward))
    # print(angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi)
    # print(forward, backward, torch.mm(forward, backward))
    return forward, backward


def jitter_data(points, jitter_range):
    """
    :param points: initial points
    :param jitter_range: (sigma, range_high)
    :return: points
    """
    sigma, clip_range = jitter_range
    jitter_data = sigma * torch.randn(points.shape).double()
    jitter_data = torch.clamp(jitter_data, -clip_range, clip_range)
    return points + jitter_data


class ToTensor(object):
    """
    Convert Points to Tensor
    You should use it before Affine
    """

    def __call__(self, sample):
        """
        Args:
            points: NumpyArray
            landmarks: NumpyArray
        Returns:
            points: Tensor
            landmarks: Tensor
        """
        sample['points'] = torch.from_numpy(sample['points'])
        sample['landmarks'] = torch.from_numpy(sample['landmarks'])
        if 'colors' in sample.keys():
            sample['colors'] = torch.from_numpy(sample['colors']).float()
        return sample


class SamplePoints(object):
    def __init__(self, sample_points):
        """
        :param sample_points: sample from initial points
        """
        self.sample_points = sample_points

    def __call__(self, sample):
        """
        Args:
            points: Array
            landmarks: Array
        Returns:
            points: NumpyArray
            landmarks: NumpyArray
        """
        initial_points, landmarks = sample['points'], sample['landmarks']
        landmarks = np.array(landmarks).reshape(-1, 3).astype(float)
        random.shuffle(initial_points)
        points, L = [], len(initial_points)
        while len(points) < self.sample_points:
            points.extend(initial_points[:min(L, self.sample_points - len(points))])
        # print(points[l], initial_points[0], l)
        points = np.array(points).astype(float)
        sample['points'], sample['landmarks'] = points[:, :3], landmarks
        if points.shape[-1] != 3:
            sample['colors'] = points[:, 3:]
        return sample


class RandomAffine(object):
    """
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
    """

    def __init__(self, normalize=None, degrees=None, translate=None, scale=None, shear=None, mirror=False,
                 corr_list=None, jitter=None, shift=0):
        """
        :param normalize:
        :param degrees:
        :param translate:
        :param scale:
        :param shear:
        :param mirror:
        :param corr_list:
        :param jitter:
        """
        if normalize is not None:
            assert isinstance(normalize, str), "normalize must be string"
        self.normalize = normalize
        if degrees is not None:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 3, "degrees should be a single number"
            for s in degrees:
                if s < 0:
                    raise ValueError("degree values must be positive.")
        self.degrees = degrees
        if translate is not None:
            assert isinstance(translate, numbers.Number), "translate should be a single number"
            if translate < 0:
                raise ValueError("If translate is a single number, it must be positive.")
        self.translate = translate
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        if shear is not None:
            assert isinstance(shear, numbers.Number), "shear should be a single number"
            if shear < 0:
                raise ValueError("If shear is a single number, it must be positive.")
        self.shear = shear
        self.mirror = mirror  # TODO; change label
        if self.mirror:
            raise NotImplementedError('Mirror(corr_list) is not done')
        self.corr_list = corr_list  # TODO; for mirror
        if jitter is not None:  # jitter?
            assert isinstance(jitter, (tuple, list)) and len(jitter) == 2, \
                "jitter should be a list or tuple and it must be of length 2."
            for s in jitter:
                if s < 0:
                    raise ValueError("jitter values should be positive")
        self.jitter = jitter
        self.shift = shift

    def getAffileMat(self, forwardAffineMat, backwardAffineMat, percentage=1):
        if self.degrees:
            degrees = self.degrees
            for value in degrees:
                value *= percentage
            forward, backward = getAffineMatrixRotate(degrees)
            forwardAffineMat = torch.mm(forwardAffineMat, forward)
            backwardAffineMat = torch.mm(backward, backwardAffineMat)

        if self.translate:
            translate = self.translate * percentage
            forward, backward = getAffineMatrixTranslate(translate)
            forwardAffineMat = torch.mm(forwardAffineMat, forward)
            backwardAffineMat = torch.mm(backward, backwardAffineMat)

        if self.scale:
            scale = self.scale
            for value in scale:
                value = (value - 1) * percentage + value
            forward, backward = getAffineMatrixScale(scale)
            forwardAffineMat = torch.mm(forwardAffineMat, forward)
            backwardAffineMat = torch.mm(backward, backwardAffineMat)

        if self.shear:
            shear = self.shear
            shear = shear * percentage
            forward, backward = getAffineMatrixShear(shear)
            forwardAffineMat = torch.mm(forwardAffineMat, forward)
            backwardAffineMat = torch.mm(backward, backwardAffineMat)
        return forwardAffineMat, backwardAffineMat

    def __call__(self, sample):
        points, landmarks = sample['points'], sample['landmarks']
        if self.normalize:
            forward, backward = getAffineMatrixNormalize(points, self.normalize)
            forwardAffineMat = forward
            backwardAffineMat = backward

        if self.jitter:
            points = jitter_data(points, self.jitter)

        # print(points, landmarks)
        # points = applyAffineMatrix3D(points, forwardAffineMat) # affine matrix not applied
        # landmarks = applyAffineMatrix3D(landmarks, forwardAffineMat)
        sample['points'] = points
        sample['landmarks'] = landmarks
        sample['forwardAffineMat'], sample['backwardAffineMat'] = self.getAffileMat(forwardAffineMat, backwardAffineMat)
        if self.shift:
            sample['forwardAffineMatShift'], sample['backwardAffineMatShift'] = self.getAffileMat(torch.eye(4).double(),
                                                                                                  torch.eye(4).double(),
                                                                                                  self.shift)
        # print(sample['forwardAffineMat'])
        # print(sample['forwardAffineMat'], sample['forwardAffineMatShift'])
        # print(points, landmarks)
        # print('final multi:', torch.mm(forwardAffineMat, backwardAffineMat), '\n back',
        #      torch.mm(backwardAffineMat, forwardAffineMat))
        # save_points(points, landmarks, 'check.obj', 'check_result.obj')
        # exit()
        # backwardAffineMat = forwardAffineMat.inverse() is also okay
        return sample


if __name__ == '__main__':
    # points = torch.ones([100, 3]).double()
    # print(points)
    # forward, backward = getAffineMatrixTranslate(0.02)
    forward, backward = getAffineMatrixRotate((15, 15, 15))
    print(forward, backward, torch.mm(forward, backward))
    # print(jitter_data(points, [0.001, 0.005]))
