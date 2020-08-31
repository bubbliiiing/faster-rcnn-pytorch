import torch
import torch.nn as nn
from nets.vgg16 import decom_vgg16
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.classifier import VGG16RoIHead,Resnet50RoIHead
import time 
import numpy as np
class FasterRCNN(nn.Module):
    def __init__(self, num_classes, 
                mode = "training",
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2),
                feat_stride = 16,
                anchor_scales = [8, 16, 32],
                ratios = [0.5, 1, 2],
                backbone = 'vgg'
    ):
        super(FasterRCNN, self).__init__()
    
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.feat_stride = feat_stride
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16()
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode = mode
            )
            self.head = VGG16RoIHead(
                n_class=num_classes + 1,
                roi_size=7,
                spatial_scale=(1. / self.feat_stride),
                classifier=classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50()

            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode = mode
            )
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=(1. / self.feat_stride),
                classifier=classifier
            )
    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn.forward(h, img_size, scale)
            
        # print(np.shape(h))
        # print(np.shape(rois))
        # print(roi_indices)
        roi_cls_locs, roi_scores = self.head.forward(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices