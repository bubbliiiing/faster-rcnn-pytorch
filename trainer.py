from __future__ import  absolute_import
import os
import time
from collections import namedtuple
from utils.utils import AnchorTargetCreator,ProposalTargetCreator
from torch.nn import functional as F
from torch import nn
import torch as torch

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn,optimizer):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3
        self.roi_sigma = 1

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = optimizer

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        
        _, _, H, W = imgs.shape
        img_size = (H, W)

        # 获取真实框和标签
        bbox = bboxes[0]
        label = labels[0]
        
        # 获取公用特征层
        features = self.faster_rcnn.extractor(imgs)

        # 获取faster_rcnn的建议框参数
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # 获取建议框的置信度和回归系数
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        # ------------------------------------------ #
        #   建议框网络的loss
        # ------------------------------------------ #
        # 先获取建议框网络应该有的预测结果
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox.cpu().numpy(),
            anchor,
            img_size)
    
        gt_rpn_label = torch.Tensor(gt_rpn_label).long()
        gt_rpn_loc = torch.Tensor(gt_rpn_loc)

        # 计算建议框网络的loss值#
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc,
                                    gt_rpn_loc,
                                    gt_rpn_label.data,
                                    self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

                               
        # ------------------------------------------ #
        #   classifier网络的loss
        # ------------------------------------------ #
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            bbox.cpu().numpy(),
            label.cpu().numpy(),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
                                    features,
                                    sample_roi,
                                    sample_roi_index)

        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                              torch.Tensor(gt_roi_label).long()]
        gt_roi_label = torch.Tensor(gt_roi_label).long()
        gt_roi_loc = torch.Tensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
                        roi_loc.contiguous(),
                        gt_roi_loc,
                        gt_roi_label.data,
                        self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())


        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    in_weight = in_weight.cuda()
    x = x.cuda()
    t = t.cuda()
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape)
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    # smooth_l1损失函数
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # 进行标准化
    loc_loss /= ((gt_label >= 0).sum().float()+1)
    return loc_loss
