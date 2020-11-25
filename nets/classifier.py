import warnings
from collections import namedtuple
from string import Template

import torch
from torch import nn
from torch.autograd import Function
from torchvision.ops import RoIPool
import numpy as np
warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
        
    def forward(self, x, rois, roi_indices):
        roi_indices = torch.Tensor(roi_indices).float()
        rois = torch.Tensor(rois).float()
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        
        xy_indices_and_rois = indices_and_rois[:, [0, 1, 2, 3, 4]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        # 利用建议框对公用特征层进行截取
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(Resnet50RoIHead, self).__init__()
        # 获得用于分类的层
        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        # 分多少个类，包括背景
        self.n_class = n_class
        # 以VGG为backbone时，roi_size为7
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale  
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        roi_indices = torch.Tensor(roi_indices).float()
        rois = torch.Tensor(rois).float()
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
            
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        xy_indices_and_rois = indices_and_rois[:, [0, 1, 2, 3, 4]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        # 利用建议框对公用特征层进行截取
        pool = self.roi(x, indices_and_rois)
        fc7 = self.classifier(pool)
        fc7 = fc7.view(fc7.size(0), -1)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
