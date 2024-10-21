import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from models.EDSR import EDSR
from dataset import demoDataset
import numpy as np
import os
from basic_option import SimpleOptions
import pickle
import time
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

def main():
    # Initialize parser
    option = SimpleOptions()
    opt = option.parse()
    opt.isTrain = True
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_blocks = (opt.image_height // opt.block_size) * (opt.image_width // opt.block_size)
    senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir, '3_TRAINING')) if f.is_dir()]
    opt.senarios = senarios

    # Prepare data
    train_dataset = demoDataset(opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    with open(f'data_stats/div2k/stats_qp{opt.qp}.pkl', 'rb') as f:
        stats = pickle.load(f)

    dct_min = torch.from_numpy(stats['dct_input']['min'][None, :, None, None]).float().to(opt.device)
    dct_max = torch.from_numpy(stats['dct_input']['max'][None, :, None, None]).float().to(opt.device)

    # Initialize model, criterion, and optimizer
    model = EDSR(n_blocks, dct_max, dct_min).to(opt.device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir='./logs')  # You can change the directory

    # DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for training!")
        model = nn.DataParallel(model)

    # Training loop
    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        for se_idx in range(len(opt.senarios)):
            print('Processing:', opt.senarios[se_idx])
            opt.se_idx = se_idx
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                batch_start_time = time.time()

                left_image, right_image, left_events, right_events = inputs

                left_image = left_image.to(opt.device)
                right_image = right_image.to(opt.device)
                left_events = left_events.to(opt.device)
                right_events = right_events.to(opt.device)
                labels = labels.to(opt.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(opt, left_image, left_events, right_image, right_events)
                loss = criterion(output, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Log the loss to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

                # Calculate the batch time
                batch_time = time.time() - batch_start_time
                
                if i % 10 == 9:  # Log every 10 batches
                    print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.3f}, Batch Time: {batch_time:.3f} sec")
                    running_loss = 0.0

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.3f} seconds")

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            model_dir = './save_models'
            model_name = f"model_epoch_{epoch + 1}_batch_{opt.batch_size}_insert_{opt.skip_number}.pth"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, model_name)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    model_name = f"model_epoch_{opt.num_epochs}_batch_{opt.batch_size}_insert_{opt.skip_number}.pth"
    model_save_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
