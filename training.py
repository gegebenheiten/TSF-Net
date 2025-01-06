import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
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
from torch.utils.tensorboard import SummaryWriter  

def mse(labels, output):
    return torch.mean((labels - output) ** 2)

def mae(labels, output):
    return torch.mean(torch.abs(labels - output))

def psnr(labels, output, max_value=1.0):
    mse_value = mse(labels, output)
    if mse_value == 0:
        return float('inf')  
    return 10 * torch.log10(max_value ** 2 / mse_value)

def validate(opt, model, val_loader, writer, epoch):
    model.eval()  
    step = 0
    val_loss = 0.0
    ssim_sum = 0.0
    ssim_loss_sum = 0.0
    L1_loss_sum = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    psnr_sum = 0.0
    with torch.no_grad():  
        for loader in val_loader:
            for i, (left_image, right_image, gt_image, left_voxel_grid, right_voxel_grid, name) in enumerate(loader):
                left_image = left_image.to(opt.device)
                right_image = right_image.to(opt.device)
                gt_image = gt_image.to(opt.device)
                left_voxel_grid = left_voxel_grid.to(opt.device)
                right_voxel_grid = right_voxel_grid.to(opt.device)

                # Forward pass
                output, _ = model(opt, left_image, left_voxel_grid, right_image, right_voxel_grid)

                # Calculate loss
                L1_loss = nn.L1Loss()(output, gt_image)
                ssim_error = ssim( gt_image, output, data_range=1.0, size_average=True)
                ssim_loss = 1 - ssim_error
                loss = L1_loss * 0.15 + ssim_loss * 0.85

                mean_square_error = mse(gt_image, output)
                mean_absolute_error = mae(gt_image, output)
                psnr_error = psnr(gt_image, output)

                val_loss += loss.item()
                ssim_sum += ssim_error.item()
                L1_loss_sum += L1_loss.item()
                ssim_loss_sum += ssim_loss.item()
                mse_sum += mean_square_error.item()
                mae_sum += mean_absolute_error.item()
                psnr_sum += psnr_error.item()
                
                step+=1
                    
    # Average validation loss and SSIM
    avg_val_loss = val_loss / step
    avg_L1_loss = L1_loss_sum / step
    avg_ssim_loss = ssim_loss_sum / step
    avg_ssim = ssim_sum / step
    avg_mse_error = mse_sum / step
    avg_mae_error = mae_sum / step
    avg_psnr = psnr_sum / step

    # Log validation metrics
    writer.add_scalar('validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('validation/L1_loss', avg_L1_loss, epoch)
    writer.add_scalar('validation/SSIM_loss', avg_ssim_loss, epoch)
    writer.add_scalar('validation/Structural Similarity Index', avg_ssim, epoch)
    writer.add_scalar('validation/Mean Squared Error', avg_mse_error, epoch)
    writer.add_scalar('validation/Mean Absolute Error', avg_mae_error, epoch)
    writer.add_scalar('validation/PSNR', avg_psnr, epoch)
    # if epoch % 10 == 0:
    #     writer.add_images('Model output/Validation', output_denorm, epoch)
    print('--------Validation-----------')
    print(f'Validation Loss: {avg_val_loss:.4f}, Average L1_loss:{avg_L1_loss:.4f}, Average SSIM loss: {avg_ssim_loss:.4f}, Average SSIM: {avg_ssim:.4f}')
    print('--------Validation-----------')


def main():
    option = SimpleOptions()
    opt = option.parse()
    opt.isTrain = True
    opt.isValidate = True
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_blocks = (opt.image_height // opt.block_size) * (opt.image_width // opt.block_size)
    senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir, 'train')) if f.is_dir()]
    opt.senarios = senarios

    val_senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir, 'val')) if f.is_dir()]
    opt.val_senarios = val_senarios

    # Prepare training data
    train_dataset = [HSERGBDataset(opt.data_root_dir, 'train', k, opt.skip_number, opt.nmb_bins) for k in senarios]
    train_loader = [DataLoader(train_dataset[k], batch_size=opt.batch_size, shuffle=True, pin_memory=False, num_workers=5) for k in range(len(train_dataset))]
    # Prepare validation data
    if opt.isValidate == True:
        val_dataset = [HSERGBDataset(opt.data_root_dir, 'val', k, opt.skip_number, opt.nmb_bins) for k in val_senarios]
        val_loader = [DataLoader(val_dataset[k], batch_size=1, shuffle=False, pin_memory=False, num_workers=4) for k in range(len(val_dataset))]
    
   
    with open(f'data_stats/div2k/stats_qp{opt.qp}.pkl', 'rb') as f:
        stats = pickle.load(f)

    dct_min = torch.from_numpy(stats['dct_input']['min'][None, :, None, None]).float().to(opt.device)
    dct_max = torch.from_numpy(stats['dct_input']['max'][None, :, None, None]).float().to(opt.device)
    # pdb.set_trace()
    model = EDSR(n_blocks, dct_max, dct_min).to(opt.device)
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {num_params}')

    log_dir = f'./logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}_batch_{opt.batch_size}_insert_{opt.skip_number}'
    writer = SummaryWriter(log_dir=log_dir)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for training!")
        model = nn.DataParallel(model)

    
    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        step = 0
        train_loss = 0.0
        L1_loss_sum = 0.0
        ssim_loss_sum = 0.0
        mse_sum = 0.0
        mae_sum = 0.0
        ssim_sum = 0.0
        psnr_sum = 0.0
        for loader in train_loader:
            for i, (left_image, right_image, gt_image, left_voxel_grid, right_voxel_grid, name) in enumerate(loader):
                left_image = left_image.to(opt.device)
                right_image = right_image.to(opt.device)
                gt_image = gt_image.to(opt.device)
                left_voxel_grid = left_voxel_grid.to(opt.device)
                right_voxel_grid = right_voxel_grid.to(opt.device)
                batch_start_time = time.time()

                optimizer.zero_grad()
                output, _ = model(opt, left_image, left_voxel_grid, right_image, right_voxel_grid)

                L1_loss = nn.L1Loss()(output, gt_image)
                ssim_error = ssim(gt_image, output, data_range=1.0, size_average=True)
                ssim_loss = 1 - ssim_error
                loss = L1_loss * 0.15 + ssim_loss * 0.85
                
                loss.backward()
                optimizer.step()

                mean_square_error = mse(gt_image, output)
                mean_absolute_error = mae(gt_image, output)
                psnr_error = psnr(gt_image, output)

                train_loss += loss.item()
                L1_loss_sum += L1_loss.item()
                ssim_loss_sum += ssim_loss.item()
                mse_sum += mean_square_error.item()
                mae_sum += mean_absolute_error.item()
                ssim_sum += ssim_error.item()
                psnr_sum += psnr_error.item()

                # Log the loss and metrics to TensorBoard
                batch_time = time.time() - batch_start_time
                print(f"[Epoch {epoch + 1}, Senario:{name[0].split('/')[-3]}, Batch {i + 1}] Loss: {loss.item():.3f}, L1_loss:{L1_loss.item():.3f}, ssim_loss:{ssim_loss.item():.3f}, ssim:{ssim_error.item():.3f}, Batch Time: {batch_time:.3f} sec")

                step+=1

            # torch.cuda.empty_cache()
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        # writer.add_images('Model Output', output_denorm, epoch)
        # writer.add_images('Ground Truth', labels_denorm, epoch)
        avg_train_loss = train_loss / step
        avg_L1_loss = L1_loss_sum / step
        avg_ssim_loss = ssim_loss_sum / step
        avg_mse_error = mse_sum / step
        avg_mae_error = mae_sum / step
        avg_ssim = ssim_sum / step
        avg_psnr = psnr_sum / step
        writer.add_scalar('train/Loss', avg_train_loss, epoch)
        writer.add_scalar('train/L1_loss', avg_L1_loss, epoch)
        writer.add_scalar('train/SSIM_loss', avg_ssim_loss, epoch)
        writer.add_scalar('train/SSIM', avg_ssim, epoch)
        writer.add_scalar('train/Mean Squared Error', avg_mse_error, epoch)
        writer.add_scalar('train/Mean Absolute Error', avg_mae_error, epoch)
        writer.add_scalar('train/PSNR', avg_psnr, epoch)

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

if __name__ == '__main__':
    main()