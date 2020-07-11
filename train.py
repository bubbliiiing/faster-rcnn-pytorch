from nets.frcnn import FasterRCNN
from nets.frcnn_training import Generator
from torch.autograd import Variable
from trainer import FasterRCNNTrainer
import time
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

def fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch):
    train_util = FasterRCNNTrainer(net,optimizer)
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_toal_loss = 0
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_size:
            break
        start_time = time.time()
        imgs,boxes,labels = batch[0], batch[1], batch[2]

        with torch.no_grad():
            imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
            boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
            labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
        losses = train_util.train_step(imgs, boxes, labels, 1)
        rpn_loc, rpn_cls, roi_loc, roi_cls, total = losses
        total_loss += total
        rpn_loc_loss += rpn_loc
        rpn_cls_loss += rpn_cls
        roi_loc_loss += roi_loc
        roi_cls_loss += roi_cls

        waste_time = time.time() - start_time
        print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || total_loss: %.4f|| rpn_loc_loss: %.4f || rpn_cls_loss: %.4f || roi_loc_loss: %.4f || roi_cls_loss: %.4f || %.4fs/step' \
            % (total_loss/(iteration+1), rpn_loc_loss/(iteration+1),rpn_cls_loss/(iteration+1),roi_loc_loss/(iteration+1),roi_cls_loss/(iteration+1),waste_time))

    print('Start Validation')
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_size_val:
            break
        imgs,boxes,labels = batch[0], batch[1], batch[2]
        with torch.no_grad():
            imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
            boxes = Variable(torch.from_numpy(boxes).type(torch.FloatTensor)).cuda()
            labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor)).cuda()

            train_util.optimizer.zero_grad()
            losses = train_util.forward(imgs, boxes, labels, 1)
            _,_,_,_, val_total = losses
            val_toal_loss += val_total
    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    # 参数初始化
    annotation_path = '2007_train.txt'
    EPOCH_LENGTH = 2000
    NUM_CLASSES = 20
    IMAGE_SHAPE = [600,600,3]
    BACKBONE = "resnet50"
    model = FasterRCNN(NUM_CLASSES,backbone=BACKBONE).cuda()

    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model_data/voc_weights_resnet.pth", map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    cudnn.benchmark = True

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 25
        
        optimizer = optim.Adam(model.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        gen = Generator(lines[:num_train],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
        gen_val = Generator(lines[num_train:],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
                        
        epoch_size = EPOCH_LENGTH
        epoch_size_val = int(EPOCH_LENGTH/10)
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.extractor.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch)
            lr_scheduler.step()

    if True:
        lr = 1e-5
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50
        optimizer = optim.Adam(model.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        gen = Generator(lines[:num_train],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
        gen_val = Generator(lines[num_train:],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
                        
        epoch_size = EPOCH_LENGTH
        epoch_size_val = int(EPOCH_LENGTH/10)
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.extractor.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch)
            lr_scheduler.step()
