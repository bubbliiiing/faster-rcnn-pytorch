from collections import namedtuple
from string import Template
import cupy, torch
import cupy as cp
import torch
from torch import nn
from torch.autograd import Function
from utils.roi_cupy import kernel_backward, kernel_forward

import warnings
warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        # 获得用于分类的层
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        # 分多少个类，包括背景
        self.n_class = n_class
        # 以VGG为backbone时，roi_size为7
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale  
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        roi_indices = torch.Tensor(roi_indices).cuda().float()
        rois = torch.Tensor(rois).cuda().float()
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
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        roi_indices = torch.Tensor(roi_indices).cuda().float()
        rois = torch.Tensor(rois).cuda().float()
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

Stream = namedtuple('Stream', ['ptr'])

def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K

class RoI(Function):
    def __init__(self, outh, outw, spatial_scale):
        self.forward_fn = load_kernel('roi_forward', kernel_forward)
        self.backward_fn = load_kernel('roi_backward', kernel_backward)
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale
    
    def forward(self, x, rois):
        # NOTE: MAKE SURE input is contiguous too
        x = x.contiguous()
        rois = rois.contiguous()
        self.in_size = B, C, H, W = x.size()
        self.N = N = rois.size(0)
        output = torch.zeros(N, C, self.outh, self.outw).cuda()
        self.argmax_data = torch.zeros(N, C, self.outh, self.outw).int().cuda()
        self.rois = rois
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.spatial_scale, C, H, W,
                self.outh, self.outw,
                output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args,
                        block=(CUDA_NUM_THREADS, 1, 1),
                        grid=(GET_BLOCKS(output.numel()), 1, 1),
                        stream=stream)
        return output
    def backward(self, grad_output):
        ##NOTE: IMPORTANT CONTIGUOUS
        # TODO: input
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = torch.zeros(self.in_size).cuda()
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.rois.data_ptr(),
                grad_input.data_ptr(),
                self.N, self.spatial_scale, C, H, W, self.outh, self.outw,
                grad_input.numel()]
        self.backward_fn(args=args,
                         block=(CUDA_NUM_THREADS, 1, 1),
                         grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
                         stream=stream
                         )
        return grad_input, None


class RoIPooling2D(torch.nn.Module):

    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)

