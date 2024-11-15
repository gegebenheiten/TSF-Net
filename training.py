import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.EDSR import EDSR
from dataset import demoDataset
import numpy as np
from skimage.metrics import structural_similarity as ssim
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

def calculate_ssim(labels, output):
    labels_np = labels.detach().cpu().numpy()  
    output_np = output.detach().cpu().numpy()  

    ssim_values = []
    for i in range(labels_np.shape[0]):  
        ssim_value = ssim(labels_np[i].transpose(1, 2, 0), 
                          output_np[i].transpose(1, 2, 0), 
                          multichannel=True, 
                          data_range=1.0,
                          channel_axis=2)
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)

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
    with torch.no_grad():  
        for se_idx in range(len(opt.val_senarios)):
            print('Processing:', opt.val_senarios[se_idx])
            opt.se_idx = se_idx
            for inputs, labels in val_loader:
                left_image, right_image, left_events, right_events = inputs

                left_image = left_image.to(opt.device)
                right_image = right_image.to(opt.device)
                left_events = left_events.to(opt.device)
                right_events = right_events.to(opt.device)
                labels = labels.to(opt.device)

                # Forward pass
                output, residue = model(opt, left_image, left_events, right_image, right_events)

                # Calculate loss
                output_denorm = (output + 1) / 2
                labels_denorm = (labels + 1) / 2
                L1_loss = nn.L1Loss()(output_denorm, labels_denorm)
                ssim_error = calculate_ssim(labels_denorm, output_denorm)
                loss = L1_loss * 0.15 + ssim_error * 0.85
                val_loss += loss.item()
                ssim_sum += ssim_error

                step+=1

    # Average validation loss and SSIM
    avg_val_loss = val_loss / step
    avg_ssim = ssim_sum / step
    residue_denorm = (residue + 1) / 2

    # Log validation metrics
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Metrics/Structural Similarity Index (Validation)', avg_ssim, epoch)
    writer.add_images('Model Residue(validation)', residue_denorm, epoch)
    print(f'Validation Loss: {avg_val_loss:.4f}, Average SSIM: {avg_ssim:.4f}')

def main():
    option = SimpleOptions()
    opt = option.parse()
    opt.isTrain = True
    opt.isValidate = True
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_blocks = (opt.image_height // opt.block_size) * (opt.image_width // opt.block_size)
    senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir, '3_TRAINING')) if f.is_dir()]
    opt.senarios = senarios

    val_senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir, '2_VALIDATION')) if f.is_dir()]
    opt.val_senarios = val_senarios

    # Prepare training data
    train_dataset = demoDataset(opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    if opt.isValidate == True:
        val_dataset = demoDataset(opt)  
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

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

    log_dir = f'./logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=log_dir)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for training!")
        model = nn.DataParallel(model)

    step = 0
    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        for se_idx in range(len(opt.senarios)):
            print('Processing:', opt.senarios[se_idx])
            opt.se_idx = se_idx
            running_loss = 0.0
            train_loss = 0.0
            mse_sum = 0.0
            mae_sum = 0.0
            ssim_sum = 0.0
            psnr_sum = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                batch_start_time = time.time()

                left_image, right_image, left_events, right_events = inputs

                left_image = left_image.to(opt.device)
                right_image = right_image.to(opt.device)
                left_events = left_events.to(opt.device)
                right_events = right_events.to(opt.device)
                labels = labels.to(opt.device)

                optimizer.zero_grad()
                output, residue = model(opt, left_image, left_events, right_image, right_events)

                output_denorm = (output + 1) / 2  # Denormalize to 0-1
                labels_denorm = (labels + 1) / 2
                L1_loss = nn.L1Loss()(output_denorm, labels_denorm)
                ssim_error = calculate_ssim(labels_denorm, output_denorm)
                # loss = L1_loss + (1- ssim_error)
                
                loss.backward()
                optimizer.step()

                mean_square_error = mse(labels_denorm, output_denorm)
                mean_absolute_error = mae(labels_denorm, output_denorm)
                psnr_error = psnr(labels_denorm, output_denorm)

                train_loss += loss.item()
                mse_sum += mean_square_error
                mae_sum += mean_absolute_error
                ssim_sum += ssim_error
                psnr_sum += psnr_error
                
                running_loss += loss.item()

                # Log the loss and metrics to TensorBoard
                batch_time = time.time() - batch_start_time
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.3f}, Batch Time: {batch_time:.3f} sec")
                running_loss = 0.0

                step+=1

        # writer.add_images('Model Output', output_denorm, epoch)
        # writer.add_images('Ground Truth', labels_denorm, epoch)
        avg_train_loss = train_loss / step
        avg_mse_error = mse_sum / step
        avg_mae_error = mae_sum / step
        avg_ssim = ssim_sum / step
        avg_psnr = psnr_sum / step
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Metrics/Mean Squared Error', avg_mse_error, epoch)
        writer.add_scalar('Metrics/Mean Absolute Error', avg_mae_error, epoch)
        writer.add_scalar('Metrics/Structural Similarity Index', avg_ssim, epoch)
        writer.add_scalar('Metrics/Peak Signal-to-Noise Ratio', avg_psnr, epoch)

        if epoch % 10 == 0:
            residue_denorm = (residue + 1) / 2
            writer.add_images('Model Output', output_denorm, epoch)
            writer.add_images('Ground Truth', labels_denorm, epoch)
            writer.add_images('Model Residue', residue_denorm, epoch)

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
