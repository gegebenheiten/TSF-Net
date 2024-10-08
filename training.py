import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from models.EDSR import EDSR
from dataset import demoDataset
import numpy  as np
import os
from basic_option import SimpleOptions
import pickle
import time
def main():
    # initialize parser
    option = SimpleOptions()
    opt = option.parse()
    opt.isTrain = True
    opt.device = torch.device(opt.gpu_ids)

    n_blocks = (256 // opt.block_size) * (256 // opt.block_size)
    senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir,'3_TRAINING')) if f.is_dir()]
    opt.senarios = senarios

    #opt.senarios  = ['may29_handheld_04']
    # Prepare data
    train_dataset = demoDataset(opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False,num_workers=0)

    # Initialize model, criterion, and optimizerç
    model = EDSR().to(opt.device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

    with open(f'/home/nwn9209/TSF-Net/data_stats/div2k/stats_qp{opt.qp}.pkl', 'rb') as f:
        stats = pickle.load(f)

    dct_min = torch.from_numpy(stats['dct_input']['min'][None, :, None, None]).float().to(opt.device)
    dct_max = torch.from_numpy(stats['dct_input']['max'][None, :, None, None]).float().to(opt.device)

    # Training loop
    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()  
        for se_idx in range(len(opt.senarios)):
            print('processing: ',opt.senarios[se_idx] )
            opt.se_idx = se_idx
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                batch_start_time = time.time()

                left_image, right_image, left_events, right_events = inputs

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(opt, left_image, left_events, right_image, right_events, n_blocks, dct_max, dct_min)
                loss = criterion(output, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # calculate the batch time
                batch_time = time.time() - batch_start_time
                
                # if i % 10 == 9:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 10:.3f}, Batch Time: {batch_time:.3f} sec")
                running_loss = 0.0

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.3f} seconds")

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            model_dir = './save_models'
            model_name  = f"model_epoch_{epoch+1}.pth"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir,exist_ok=True)
            model_save_path = os.path.join(model_dir,model_name)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    model_name  = f"model_epoch_{opt.num_epochs}.pth"
    model_save_path = os.path.join(model_dir,model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
if __name__ == '__main__':
    #mp.set_start_method('spawn')
    main()