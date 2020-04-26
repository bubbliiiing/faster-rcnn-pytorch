import torch
import cupy as cp
import numpy as np
from torch.nn import functional as F
def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def nms(detections_class,nms_thres=0.7):
    max_detections = []
    while np.shape(detections_class)[0]:
        # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        max_detections.append(np.expand_dims(detections_class[0],0))
        if len(detections_class) == 1:
            break
        ious = bbox_iou(max_detections[-1][:,:4], detections_class[1:,:4])[0]
        detections_class = detections_class[1:][ious < nms_thres]
    if len(max_detections)==0:
        return []
    max_detections = np.concatenate(max_detections,axis=0)
    return max_detections

class DecodeBox():
    def __init__(self, std, mean, num_classes):
        self.std = std
        self.mean = mean
        self.num_classes = num_classes+1    

    def forward(self, roi_cls_locs, roi_scores, rois, height, width, score_thresh):

        rois = torch.Tensor(rois)

        roi_cls_loc = (roi_cls_locs * self.std + self.mean)
        roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])
        roi = rois.view((-1, 1, 4)).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox((roi.cpu().detach().numpy()).reshape((-1, 4)),
                            (roi_cls_loc.cpu().detach().numpy()).reshape((-1, 4)))
        cls_bbox = torch.Tensor(cls_bbox)
        cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])
        # clip bounding box
        cls_bbox[:, 0] = (cls_bbox[:, 0]).clamp(min=0, max=width)
        cls_bbox[:, 2] = (cls_bbox[:, 2]).clamp(min=0, max=width)
        cls_bbox[:, 1] = (cls_bbox[:, 1]).clamp(min=0, max=height)
        cls_bbox[:, 3] = (cls_bbox[:, 3]).clamp(min=0, max=height)

        prob = F.softmax(torch.tensor(roi_scores), dim=1)
        raw_cls_bbox = cls_bbox.cpu().numpy()
        raw_prob = prob.cpu().numpy()

        outputs = []
        for l in range(1, self.num_classes):
            cls_bbox_l = raw_cls_bbox[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            if len(prob_l) == 0:
                continue
            label = np.ones_like(prob_l)*(l-1)
            detections_class = np.concatenate([cls_bbox_l,np.expand_dims(prob_l,axis=-1),np.expand_dims(label,axis=-1)],axis=-1)
            
            prob_l_index = np.argsort(prob_l)[::-1]
            detections_class = detections_class[prob_l_index]
            nms_out = nms(detections_class,0.3)
            if outputs==[]:
                outputs = nms_out
            else:
                outputs = np.concatenate([outputs,nms_out],axis=0)
        return outputs

class ProposalTargetCreator(object):
    def __init__(self,n_sample=128,
                 pos_ratio=0.5, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape

        # 计算正样本数量
        
        roi = np.concatenate((roi, bbox), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # 真实框的标签要+1因为有背景的存在
        gt_roi_label = label[gt_assignment] + 1

        # 找到大于门限的真实框的索引
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # 正负样本的平衡，满足建议框和真实框重合程度小于neg_iou_thresh_hi大于neg_iou_thresh_lo作为负样本
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        if neg_index.size > 0:
            try:
                neg_index = np.random.choice(
                    neg_index, size=self.n_sample - pos_roi_per_this_image, replace=False)
            except:
                neg_index = np.random.choice(
                    neg_index, size=self.n_sample - pos_roi_per_this_image, replace=True)

        # 取出这些框对应的标签
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        # 找到
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox)
        # 利用先验框和其对应的真实框进行编码
        loc = bbox2loc(anchor, bbox[argmax_ious])

        return loc, label

    def _create_label(self, anchor, bbox):
        # 1是正样本，0是负样本，-1忽略
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # argmax_ious为每个先验框对应的最大的真实框的序号
        # max_ious为每个真实框对应的最大的真实框的iou
        # gt_argmax_ious为每一个真实框对应的最大的先验框的序号
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox)

        # 如果小于门限函数则设置为负样本
        label[max_ious < self.neg_iou_thresh] = 0

        # 每个真实框至少对应一个先验框
        label[gt_argmax_ious] = 1
        
        # 如果大于门限函数则设置为正样本
        label[max_ious >= self.pos_iou_thresh] = 1

        # 判断正样本数量是否大于128，如果大于的话则去掉一些
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox):
        # 计算所有
        ious = bbox_iou(anchor, bbox)
        # 行是先验框，列是真实框
        argmax_ious = ious.argmax(axis=1)
        # 找出每一个先验框对应真实框最大的iou
        max_ious = ious[np.arange(len(anchor)), argmax_ious]
        # 行是先验框，列是真实框
        gt_argmax_ious = ious.argmax(axis=0)
        # 找到每一个真实框对应的先验框最大的iou
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # 每一个真实框对应的最大的先验框的序号
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious
