'''

'''
import random
import argparse
import os
import shutil
import time
import copy
import numpy as np
import glob2
import pickle
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from options import args_parser
from models.EDSR import EDSR

from dataloader import ImagePatchLoader
from utility import get_psnr, save_fig, count_parameters



#------------------------------------------------------------------------------

best_psnr = 0
nbit = 10
qp = 42

#------------------------------------------------------------------------------

def main():
    global args, best_psnr
    args = args_parser()
    args.device = torch.device("cuda:0")
    
    args.epochs = 150
    args.qp = qp
    args.patch_size = 256
    args.block_size = 4
    args.num_patches_per_frame = 50
    
    args.div2kdata = 'path/to/training_dataset/div2k/ExtractedFrames'
    frame_list = os.listdir(os.path.join(args.div2kdata, 'Uncompressed'))
    frame_list.sort()
    
    frame_size = [np.array(np.load( os.path.join(args.div2kdata, 'Uncompressed', frame) ).shape[1:])*2 for frame in frame_list]
    
    train_list = [[frame, size] for (frame, size) in zip(frame_list[:875], frame_size[:875])]
    valid_list = [[frame, size] for (frame, size) in zip(frame_list[875:], frame_size[875:])]
    
    val_loader = torch.utils.data.DataLoader(
            ImagePatchLoader(args, valid_list, nbit, transforms=False, train=False),
            batch_size=16, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    with open(f'data_stats/div2k/stats_qp{qp}.pkl','rb') as f:
        stats = pickle.load(f)
    
    model = EDSR(); num_params = count_parameters(model); print(f'no. of params: {num_params} \n')
    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids).to(args.device)
    
    #optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=args.weight_decay, amsgrad=False)
    #scheduler = lrs.MultiStepLR(optim, milestones=[], gamma=args.gamma)
    
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    
    criterion = nn.L1Loss()
    
    #args.save_dir = f'/scratch/biren/save_models/LearnedInLoopFIlter/InlpDis_NpointDCT_SameSizeMSTFusionAllBlocks_SlimTall_16_largerpatch_noDCT/qp-{qp}'
    args.save_dir = f'save_temp/qp-{qp}'
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    n_iter = 0
    # optionally resume from a checkpoint
    if True: #args.resume:
        resume_epoch = 117
        resume_path = args.save_dir + f'/epoch_{resume_epoch}/checkpoint_{resume_epoch}.pth.tar'
        chkpt = torch.load(resume_path, map_location='cuda:0')
        model.load_state_dict(chkpt['model'])
        n_iter = chkpt['iter']
        args.start_epoch = resume_epoch + 1
        for _ in range(0, args.start_epoch):
            optim.zero_grad()
            optim.step()
            scheduler.step()
        
    writer = SummaryWriter(args.log_file + f'/qp-{args.qp}')
    for epoch in range(args.start_epoch, args.epochs):
        
        random.shuffle(train_list)
        # train_list = train_list[:10]
        
        # create batch of frames
        frame_batch = [train_list[i: i+4] if (i+4)<len(train_list) else train_list[i:] for i in range(0, len(train_list), 4)]
        step = 0
        for frames in frame_batch:
            
            train_loader = torch.utils.data.DataLoader(
                ImagePatchLoader(args, frames, nbit, transforms=True, train=True),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            
            # train for one batch of images
            step = train(args, train_loader, stats, model, optim, scheduler, criterion, epoch, writer, step)
        n_iter += step
        
        # evaluate on validation set
        validate(args, val_loader, stats, model, criterion, epoch, writer)
        
        save_checkpoint(args.save_dir, model, n_iter, epoch)
        writer.add_scalar('train/learning-rate', scheduler.get_last_lr()[0], epoch)  #optim.param_groups[0]['lr']
        scheduler.step()
        
        
#------------------------------------------------------------------------------

def train(args, train_loader, stats, model, optim, scheduler, criterion, epoch, writer, step):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    n_blocks = (args.patch_size//args.block_size)**2
    
    #dct_mean = torch.from_numpy(stats['dct_input']['mean'][None,:, None, None]).float().to(args.device) 
    #dct_var = torch.from_numpy(stats['dct_input']['var'][None,:, None, None]).float().to(args.device) 
    dct_min = torch.from_numpy(stats['dct_input']['min'][None,:, None, None]).float().to(args.device) 
    dct_max = torch.from_numpy(stats['dct_input']['max'][None,:, None, None]).float().to(args.device) 
    
    #stats_params(stats)
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, partition, prediction, target, _) in enumerate(train_loader):
    
        # measure data loading time
        data_time.update(time.time() - end)
        
        image = image.to(args.device)
        partition = partition.to(args.device) 
        prediction = prediction.to(args.device)
        target = target.to(args.device)
        
        # compute output
        output = model(args, image, partition, prediction, n_blocks, dct_max, dct_min)
        loss = criterion(output, target)
        
        optim.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optim.step()
        
        
        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            writer.add_scalar('train/loss', losses.val, step)
            
            print(f'Epoch: [{epoch}] [{step}/{int(np.ceil(args.num_patches_per_frame*875/args.batch_size))}], ' +  
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}), ' +
                  f'Data: {data_time.val:.3f} ({data_time.avg:.3f}), ' + 
                  f'TrainLoss: {losses.val:.3f} ({losses.avg:.3f})')
        step += 1
    
    return step

#------------------------------------------------------------------------------

def validate(args, val_loader, stats, model, criterion, epoch, writer):
    """
    Run evaluation
    """

    #batch_time = AverageMeter()
    losses = AverageMeter()
    psnr1 = AverageMeter()
    psnr2 = AverageMeter()
    
    n_blocks = (args.patch_size//args.block_size)**2
    
    #dct_mean = torch.from_numpy(stats['dct_input']['mean'][None,:, None, None]).float().to(args.device) 
    #dct_var = torch.from_numpy(stats['dct_input']['var'][None,:, None, None]).float().to(args.device) 
    dct_min = torch.from_numpy(stats['dct_input']['min'][None,:, None, None]).float().to(args.device) 
    dct_max = torch.from_numpy(stats['dct_input']['max'][None,:, None, None]).float().to(args.device) 

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (image, partition, prediction, target, image_vvc) in enumerate(val_loader):

            image = image.to(args.device)
            partition = partition.to(args.device) 
            prediction = prediction.to(args.device)
            target = target.to(args.device)
            
            # compute output
            output = model(args, image, partition, prediction, n_blocks, dct_max, dct_min)
            loss = criterion(output, target)
    
            output = output.float()
            loss = loss.float()
    
            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            
            if i%10==0:
                output = output.cpu().numpy() 
                image_vvc = image_vvc.numpy()
                target = target.cpu().numpy()
                for (predImage, vvcImage, gtImage) in zip(output, image_vvc, target):
                    psnr_valid = get_psnr(gtImage[0], vvcImage)
                    psnr_pred = get_psnr(gtImage[0], predImage[0])
                    
                    psnr1.update(psnr_valid)
                    psnr2.update(psnr_pred)
                
        writer.add_scalar('valid/loss', losses.avg, epoch)
            
        print(f'Epoch: [{epoch}] [{i}/{len(val_loader)}], ' +  
              f'ValidLoss: {losses.val:.3f} ({losses.avg:.3f})')
    
    writer.add_scalar('valid/PSNR-VVC', psnr1.avg, epoch)
    writer.add_scalar('valid/PSNR-NN', psnr2.avg, epoch)
    print('PSNR-VVC: {psnr1.avg:.3f}, PSNR-NN: {psnr2.avg:.3f}'.format(psnr1=psnr1, psnr2=psnr2))

    return None

#------------------------------------------------------------------------------

def save_checkpoint(dir_path, model, n_iter, epoch):
    """
    Save the training model
    """
    save_dir = os.path.join(dir_path, 'epoch_{}'.format(epoch))
    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir)
    state = {'epoch': epoch, 'iter': n_iter, 'model': model.state_dict()}
    filename = os.path.join(save_dir, 'checkpoint_{}.pth.tar'.format(epoch))
    torch.save(state, filename)
    
#------------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#------------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
