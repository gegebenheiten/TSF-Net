import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from models.EDSR import EDSR
from dataset import HSERGBDataset
import numpy as np
from pytorch_msssim import ssim
import os
from basic_option import SimpleOptions
import pickle
import time
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter  

def mse(labels, output):
    return torch.mean((labels - output) ** 2)

def mae(labels, output):
    return torch.mean(torch.abs(labels - output))

def psnr(labels, output, max_value=255.0):
    mse_value = mse(labels, output)
    if mse_value == 0:
        return float('inf')  
    return 10 * torch.log10(max_value ** 2 / mse_value)

def init_process(rank, size, fn, backend='nccl'):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def validate(opt, model, val_loader, writer, epoch):
    model.eval()  
    step = 0
    val_loss = 0.0
    ssim_sum = 0.0
    L1_loss_sum = 0.0
    with torch.no_grad():  
        for loader in val_loader:
            for i, (events_forward, events_backward, left_image, right_image, gt_image, weight, [n_left, n_right],
                    surface, left_voxel_grid, right_voxel_grid, name) in enumerate(loader):

                    events_forward = events_forward.cuda()
                    events_backward = events_backward.cuda()
                    left_image = left_image.cuda()
                    right_image = right_image.cuda()
                    gt_image = gt_image.cuda()
                    # weight = weight.cuda()
                    # surface = surface.cuda()
                    left_voxel_grid = left_voxel_grid.cuda()
                    right_voxel_grid = right_voxel_grid.cuda()

                    # Forward pass
                    output, residue = model(opt, left_image, left_voxel_grid, right_image, right_voxel_grid)

                    # Calculate loss
                    output_denorm = (output + 1) / 2 * 255
                    labels_denorm = (gt_image + 1) / 2 * 255
                    L1_loss = nn.L1Loss()(output_denorm, labels_denorm)
                    ssim_error = ssim( labels_denorm, output_denorm, data_range=255, size_average=True)
                    loss = L1_loss * 0.15 + (1 - ssim_error)  * 0.85
                    val_loss += loss.item()
                    ssim_sum += ssim_error
                    L1_loss_sum += L1_loss.item()
                    
                    step+=1
                    
    # Average validation loss and SSIM
    avg_val_loss = val_loss / step
    avg_L1_loss = L1_loss_sum / step
    avg_ssim = ssim_sum / step

    # Log validation metrics
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Loss/validation/L1_loss', avg_L1_loss, epoch)
    writer.add_scalar('Loss/validation/Structural Similarity Index', avg_ssim, epoch)
    # if epoch % 10 == 0:
    #     writer.add_images('Model output/Validation', output_denorm, epoch)
    print('--------Validation-----------')
    print(f'Validation Loss: {avg_val_loss:.4f}, Average L1_loss:{avg_L1_loss:.4f}, Average SSIM: {avg_ssim:.4f}')
    print('--------Validation-----------')

    # Clear GPU memory after validation
    torch.cuda.empty_cache()


def train(rank, size):
    # Set the device for each process
    torch.cuda.set_device(rank)  # Ensure each process uses a different GPU
    opt = SimpleOptions().parse()
    opt.device = torch.device('cuda', rank)  # Set device for current process
    opt.isTrain = True
    opt.isValidate = True

    n_blocks = (opt.image_height // opt.block_size) * (opt.image_width // opt.block_size)
    senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir, 'train')) if f.is_dir()]
    opt.senarios = senarios

    val_senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir, 'val')) if f.is_dir()]
    opt.val_senarios = val_senarios

    # Prepare training data with DistributedSampler
    train_dataset = [HSERGBDataset(opt.data_root_dir, 'train', k, opt.skip_number, opt.nmb_bins) for k in senarios]
    train_sampler = DistributedSampler(train_dataset[rank], num_replicas=size, rank=rank)
    train_loader = DataLoader(train_dataset[rank], batch_size=opt.batch_size, sampler=train_sampler, num_workers=8)

    # Prepare validation data
    if opt.isValidate == True:
        val_dataset = [HSERGBDataset(opt.data_root_dir, 'val', k, opt.skip_number, opt.nmb_bins) for k in val_senarios]
        val_loader = [torch.utils.data.DataLoader(val_dataset[k], batch_size=1, shuffle=False, pin_memory=False, num_workers=8) for k in range(len(val_dataset))]
    
   
    with open(f'data_stats/div2k/stats_qp{opt.qp}.pkl', 'rb') as f:
        stats = pickle.load(f)

    dct_min = torch.from_numpy(stats['dct_input']['min'][None, :, None, None]).float().to(opt.device)
    dct_max = torch.from_numpy(stats['dct_input']['max'][None, :, None, None]).float().to(opt.device)

    model = EDSR(n_blocks, dct_max, dct_min).to(opt.device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {num_params}')

    log_dir = f'./logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=log_dir)

    step = 0
    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        train_loss = 0.0
        L1_loss_sum = 0.0
        mse_sum = 0.0
        mae_sum = 0.0
        ssim_sum = 0.0
        psnr_sum = 0.0

        train_sampler.set_epoch(epoch)  # Shuffle data at the start of each epoch
        
        for loader in train_loader:
            for i, (events_forward, events_backward, left_image, right_image, gt_image, weight, [n_left, n_right],
                    surface, left_voxel_grid, right_voxel_grid, name) in enumerate(loader):
                events_forward = events_forward.cuda()
                events_backward = events_backward.cuda()
                left_image = left_image.cuda()
                right_image = right_image.cuda()
                gt_image = gt_image.cuda()
                # weight = weight.cuda()
                # surface = surface.cuda()
                left_voxel_grid = left_voxel_grid.cuda()
                right_voxel_grid = right_voxel_grid.cuda()
                batch_start_time = time.time()

                optimizer.zero_grad()
                output, residue = model(opt, left_image, left_voxel_grid, right_image, right_voxel_grid)

                output_denorm = (output + 1) / 2  * 255
                labels_denorm = (gt_image + 1) / 2 * 255
                L1_loss = nn.L1Loss()(output_denorm, labels_denorm)
                ssim_error = ssim(labels_denorm, output_denorm, data_range=255, size_average=True)
                loss = L1_loss * 0.15 + (1 - ssim_error) * 0.85
                
                loss.backward()
                optimizer.step()

                mean_square_error = mse(labels_denorm, output_denorm)
                mean_absolute_error = mae(labels_denorm, output_denorm)
                psnr_error = psnr(labels_denorm, output_denorm)

                train_loss += loss.item()
                L1_loss_sum += L1_loss.item()
                mse_sum += mean_square_error
                mae_sum += mean_absolute_error
                ssim_sum += ssim_error
                psnr_sum += psnr_error

                # Log the loss and metrics to TensorBoard
                batch_time = time.time() - batch_start_time
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss:.3f}, L1_loss:{L1_loss:.3f}, ssim:{ssim_error:.3f}, Batch Time: {batch_time:.3f} sec")

                step+=1
                            
            torch.cuda.empty_cache()

        # writer.add_images('Model Output', output_denorm, epoch)
        # writer.add_images('Ground Truth', labels_denorm, epoch)
        avg_train_loss = train_loss / step
        avg_L1_loss = L1_loss_sum / step
        avg_mse_error = mse_sum / step
        avg_mae_error = mae_sum / step
        avg_ssim = ssim_sum / step
        avg_psnr = psnr_sum / step
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train/L1_loss', avg_L1_loss, epoch)
        writer.add_scalar('Loss/train/Structural Similarity Index', avg_ssim, epoch)
        writer.add_scalar('Metrics/Mean Squared Error', avg_mse_error, epoch)
        writer.add_scalar('Metrics/Mean Absolute Error', avg_mae_error, epoch)
        writer.add_scalar('Metrics/Peak Signal-to-Noise Ratio', avg_psnr, epoch)

        # if epoch % 10 == 0:
        #     # residue_denorm = (residue + 1) / 2
        #     writer.add_images('Model Output/Train', output_denorm, epoch)
        #     writer.add_images('Ground Truth', labels_denorm, epoch)
        #     # writer.add_images('Model Residue', residue_denorm, epoch)

        # Validation at the end of each epoch
        if opt.isValidate == True:
            validate(opt, model, val_loader, writer, epoch)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.3f} seconds")

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            model_dir = './save_models'
            model_name = f"model_epoch_{epoch + 1}_batch_{opt.batch_size}_insert_{opt.skip_number}_small.pth"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, model_name)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    model_name = f"model_epoch_{opt.num_epochs}_batch_{opt.batch_size}_insert_{opt.skip_number}_small.pth"
    model_save_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()

def main():
    size = torch.cuda.device_count()  # Number of GPUs
    mp.spawn(init_process, args=(size, train), nprocs=size, join=True)

if __name__ == '__main__':
    main()