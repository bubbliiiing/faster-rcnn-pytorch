from nets.frcnn import FasterRCNN
from torchsummary import summary
from PIL import Image
import numpy as np
from utils.utils import loc2bbox
import torch
from thop import profile
from thop import clever_format
import torch
model =  FasterRCNN(20,backbone="resnet50").cuda()
# model.load_state_dict(torch.load("logs/Epoch7-Total_Loss0.9575.pth"))
a = np.array(Image.open("img/street.jpg").resize([600,600]))
a = np.transpose(a,[2,0,1])
a = torch.Tensor(np.concatenate([np.expand_dims(a,0),np.expand_dims(a,0)],axis=0)).cuda()
roi_cls_locs, roi_scores, rois, roi_indices = model(a)
rois = torch.Tensor(rois)
# mean = torch.Tensor([0,0,0,0]).cuda(). \
#     repeat(21)[None]
# std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).cuda(). \
#     repeat(21)[None]

# roi_cls_loc = (roi_cls_locs * std + mean)
# roi_cls_loc = roi_cls_loc.view([-1, 21, 4])
# roi = rois.view((-1, 1, 4)).expand_as(roi_cls_loc)
# cls_bbox = loc2bbox((roi.cpu().detach().numpy()).reshape((-1, 4)),
#                     (roi_cls_loc.cpu().detach().numpy()).reshape((-1, 4)))
# cls_bbox = torch.Tensor(cls_bbox)
# cls_bbox = cls_bbox.view([-1, 21 * 4])
# print(cls_bbox)
# # clip bounding box
# cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=600)
# cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=800)


# print(roi_cls_locs,roi_scores)

# # input = torch.randn(1, 3, 600, 600).cuda()
# # flops, params = profile(model,inputs=(input,))
# # flops, params = clever_format([flops, params], "%.3f")
# # params = list(model.parameters())
# # k = 0
# # for i in params:
# #     l = 1
# #     print("该层的结构：" + str(list(i.size())))
# #     for j in i.size():
# #         l *= j
# #     print("该层参数和：" + str(l))
# #     k = k + l
# # print("总参数数量和：" + str(k))