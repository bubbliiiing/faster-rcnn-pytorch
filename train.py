from nets.frcnn import FasterRCNN
from nets.frcnn_training import Generator
from torch.autograd import Variable
from trainer import FasterRCNNTrainer
import time
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch):
    train_util = FasterRCNNTrainer(net,optimizer)
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_toal_loss = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
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
            
            pbar.set_postfix(**{'total'    : total_loss.item() / (iteration + 1), 
                                'rpn_loc'  : rpn_loc_loss.item() / (iteration + 1),  
                                'rpn_cls'  : rpn_cls_loss.item() / (iteration + 1), 
                                'roi_loc'  : roi_loc_loss.item() / (iteration + 1), 
                                'roi_cls'  : roi_cls_loss.item() / (iteration + 1), 
                                'lr'       : get_lr(optimizer)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
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
            pbar.set_postfix(**{'total_loss': val_toal_loss.item() / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

if __name__ == "__main__":
    # 参数初始化
    annotation_path = '2007_train.txt'
    NUM_CLASSES = 20
    IMAGE_SHAPE = [600,600,3]
    BACKBONE = "resnet50"
    model = FasterRCNN(NUM_CLASSES,backbone=BACKBONE).cuda()
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    model_path = r'model_data/voc_weights_resnet.pth'
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
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
    
    if True:
        lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 25
        
        optimizer = optim.SGD(model.parameters(),lr,weight_decay=5e-4,momentum=0.9)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        if Use_Data_Loader:
            train_dataset = FRCNNDataset(lines[:num_train],(IMAGE_SHAPE[0],IMAGE_SHAPE[1]))
            val_dataset   = FRCNNDataset(lines[num_train:],(IMAGE_SHAPE[0],IMAGE_SHAPE[1]))
            gen     = DataLoader(train_dataset, batch_size=1, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=1, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)
        else:
            gen     = Generator(lines[:num_train],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
            gen_val = Generator(lines[num_train:],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
                        
        epoch_size = num_train
        epoch_size_val = num_val
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.extractor.parameters():
            param.requires_grad = False

        # ------------------------------------#
        #   由于batch==1所以冻结bn层
        # ------------------------------------#
        model = model.eval()

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50
        optimizer = optim.SGD(model.parameters(),lr,weight_decay=5e-4,momentum=0.9)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

        if Use_Data_Loader:
            train_dataset = FRCNNDataset(lines[:num_train],(IMAGE_SHAPE[0],IMAGE_SHAPE[1]))
            val_dataset   = FRCNNDataset(lines[num_train:],(IMAGE_SHAPE[0],IMAGE_SHAPE[1]))
            gen     = DataLoader(train_dataset, batch_size=1, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=1, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)
        else:
            gen     = Generator(lines[:num_train],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
            gen_val = Generator(lines[num_train:],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
                        
        epoch_size = num_train
        epoch_size_val = num_val
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.extractor.parameters():
            param.requires_grad = True

        # ------------------------------------#
        #   由于batch==1所以冻结bn层
        # ------------------------------------#
        model = model.eval()

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch)
            lr_scheduler.step()
