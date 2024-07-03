import os
import sys
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from .EDSR import EDSR
#from .utility import make_optimizer
#from .my_loss import Loss

#------------------------------------------------------------------------------

class WaveletNet(object):
    def __init__(self, subband_order, args):
        super(WaveletNet, self).__init__()
        self.args = args
        self.psnr = 0
        self.epoch = 0
        self.models = {subband:EDSR(subband) for subband in subband_order} 
        # self.optims = {subband:torch.optim.SGD(self.models[subband].parameters(), args.lr, 
        #                                  momentum=args.momentum, 
        #                                  weight_decay=args.weight_decay) for subband in subband_order} 
        # self.criterions = {subband:nn.MSELoss() for subband in subband_order} 
        
        self.optims = {subband:torch.optim.Adam(self.models[subband].parameters(), 
                                                lr=args.lr, betas=args.betas, eps=args.epsilon, 
                                                weight_decay=args.weight_decay, amsgrad=False) for subband in subband_order} 
        self.schedulers = {subband:lrs.MultiStepLR(self.optims[subband], milestones=[10, 20], gamma=args.gamma) for subband in subband_order}
        self.criterions = {subband:nn.L1Loss() for subband in subband_order}
        
        
        #self.optims = {subband: make_optimizer(args, self.models[subband]) for subband in subband_order} 
        #self.criterions = {subband:Loss(args) for subband in subband_order}
        self.subband_order = subband_order
        
    # save all models in a single file
    def save_model(self, epoch):
        dir_path = os.path.join(self.args.save_dir, 'epoch_{}'.format(epoch))
        if not os.path.isdir(dir_path): 
            os.makedirs(dir_path)
        state = {**{'epoch': epoch, 'best_psnr': self.psnr}, 
                 **{f'model_{subband}': self.models[subband].state_dict() for subband in self.subband_order},
                 **{f'optimizer_{subband}' : self.optims[subband].state_dict() for subband in self.subband_order},
                 **{f'scheduler_{subband}' : self.schedulers[subband].state_dict() for subband in self.subband_order},
                 **{f'loss_{subband}' : self.criterions[subband].state_dict() for subband in self.subband_order}
                 }
        filename = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(epoch))
        torch.save(state, filename)
    
    # save all models from a single file
    def load_model(self, checkpoint=None):
        if checkpoint:
            filename = checkpoint
        else:
            dir_path = os.path.join(self.args.save_dir, 'epoch_{}'.format(self.args.start_epoch))
            filename = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(self.args.start_epoch))
        if os.path.isfile(filename):
            check_point = torch.load(filename)
            self.psnr = check_point['best_psnr']
            self.epoch = check_point['epoch']
            for subband in self.subband_order:
                self.models[subband].load_state_dict(check_point[f'model_{subband}'])
                self.optims[subband].load_state_dict(check_point[f'optimizer_{subband}'])
                self.schedulers[subband].load_state_dict(check_point[f'scheduler_{subband}'])
                self.criterions[subband].load_state_dict(check_point[f'loss_{subband}'])
        else:
            sys.exit('Error Loading Model!! Model Does not exists!!')
    
    # # save all models in a separate files
    # def save_model(self, epoch):
    #     dir_path = os.path.join(self.args.save_dir, 'epoch_{}'.format(epoch))
    #     if not os.path.isdir(dir_path): 
    #         os.makedirs(dir_path)
    #     for subband in self.subband_order:
    #         state = {'epoch': epoch, 
    #                  'best_psnr': self.psnr, 
    #                  'model': self.models[subband].state_dict(),
    #                  'optimizer' : self.optims[subband].state_dict()
    #                  }
    #         filename = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(subband))
    #         torch.save(state, filename)
            
    # # load all models from separate files
    # def load_model(self, checkpoint=None):
    #     if checkpoint:
    #         dir_path = checkpoint
    #     else:
    #         dir_path = os.path.join(self.args.save_dir, 'epoch_{}'.format(self.args.start_epoch))
    #     for subband in self.subband_order:
    #         filename = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(subband))
    #         if os.path.isfile(filename):
    #             check_point = torch.load(filename)
    #             self.models[subband].load_state_dict(check_point['model'])
    #             self.optims[subband].load_state_dict(check_point['optimizer'])
    #         else:
    #             sys.exit('Error Loading Model!! Model Does not exists!!')
    
    # # save individual models
    # def save_checkpoint(self, subband, best_psnr, epoch):
    #     dir_path = os.path.join(self.args.save_dir, 'epoch_{}'.format(epoch))
    #     if not os.path.isdir(dir_path): 
    #         os.makedirs(dir_path)
    #     state = {'epoch': epoch + 1, 'best_psnr': best_psnr, 'state_dict': self.models[subband].state_dict()}
    #     filename = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(subband))
    #     torch.save(state, filename)

#------------------------------------------------------------------------------

# class WaveletNet(nn.Module):
#     def __init__(self):
#         super(WaveletNet, self).__init__()
#         # LL, LH, HL, HH
#         self.edsrLL = EDSR('LL')
        
#     def forward(self, x):
#         x = self.edsrLL(x)
#         return x

#------------------------------------------------------------------------------

# class NNBasedInLoopFilter(object):
#     def __init__(self, args):
#         # LL, LH, HL, HH
#         self.edsrLL = EDSR('LL').cuda(0)
#         self.edsrLH = EDSR('LH').cuda(0)
#         self.edsrHL = EDSR('HL').cuda(1)
#         self.edsrHH = EDSR('HH').cuda(1)
        
#         self.criterionLL = nn.MSELoss().cuda(0)
#         self.criterionLL = nn.MSELoss().cuda(0)
#         self.criterionHL = nn.MSELoss().cuda(1)
#         self.criterionHH = nn.MSELoss().cuda(1)
        
#         self.optimizerLL = torch.optim.SGD(self.edsrLL.parameters(), args.lr, 
#                                          momentum=args.momentum, 
#                                          weight_decay=args.weight_decay)
        
#         self.optimizerLH = torch.optim.SGD(self.edsrLH.parameters(), args.lr, 
#                                          momentum=args.momentum, 
#                                          weight_decay=args.weight_decay)
        
#         self.optimizerHL = torch.optim.SGD(self.edsrHL.parameters(), args.lr, 
#                                          momentum=args.momentum, 
#                                          weight_decay=args.weight_decay)
        
#         self.optimizerHH = torch.optim.SGD(self.edsrHH.parameters(), args.lr, 
#                                          momentum=args.momentum, 
#                                          weight_decay=args.weight_decay)
        
        
    
#     def forward(self, input_):
#         outputLL = self.edsrLL(input_)
#         outputLH = self.edsrLH(input_)
        
#         input_ = input_.cuda(1)
#         outputHL = self.edsrHL(input_)
#         outputHH = self.edsrHH(input_)
        
#         return [outputLL, outputLH, outputHL, outputHH]
        
#     def criterion(self, output, target):
#         lossLL = self.criterion(output[0], target[:,0,:,:])
#         lossLH = self.criterion(output[1], target[:,1,:,:])
#         lossHL = self.criterion(output[2], target[:,2,:,:])
#         lossHH = self.criterion(output[3], target[:,3,:,:])
        
#         return [lossLL, lossLH, lossHL, lossHH]
        
#     def backward(self, loss):
#         self.optimizerLL.zero_grad()
#         loss[0].backward()
#         self.optimizerLL.step()
        
#         self.optimizerLH.zero_grad()
#         loss[1].backward()
#         self.optimizerLH.step()
        
#         self.optimizerHL.zero_grad()
#         loss[2].backward()
#         self.optimizerHL.step()
        
#         self.optimizerHH.zero_grad()
#         loss[3].backward()
#         self.optimizerHH.step()
        
#     def mode_train(self):
#         self.edsrLL().train()
#         self.edsrLH().train()
#         self.edsrHL().train()
#         self.edsrHH().train()
        
#     def mode_eval(self):
#         self.edsrLL().eval()
#         self.edsrLH().eval()
#         self.edsrHL().eval()
#         self.edsrHH().eval()
        
        
