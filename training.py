import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.EDSR import EDSR
from dataset import demoDataset
import numpy  as np
import os
from basic_potion import SimpleOptions
import pickle

# initialize parser
option = SimpleOptions()
opt = option.parse()
opt.isTrain = True
opt.skip_number = 7
opt.device = torch.device("cuda:0")
qp = 42
opt.epochs = 150
opt.qp = qp
opt.patch_size = 256
opt.block_size = 4
opt.num_patches_per_frame = 50
n_blocks = (256 // opt.block_size) * (256 // opt.block_size)

# Prepare data
train_dataset = demoDataset(opt, skip_number=opt.skip_number)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Initialize model, criterion, and optimizer
model = EDSR().to(opt.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

with open(f'data_stats/div2k/stats_qp{qp}.pkl', 'rb') as f:
    stats = pickle.load(f)

dct_min = torch.from_numpy(stats['dct_input']['min'][None, :, None, None]).float().to(opt.device)
dct_max = torch.from_numpy(stats['dct_input']['max'][None, :, None, None]).float().to(opt.device)

# Training loop
for epoch in range(opt.num_epochs):
    for se_idx in range(len(opt.senarios)):
        opt.se_idx = se_idx
        running_loss = 0.0
        for i, (left_image, right_image, labels, left_events, right_events) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(opt, left_image, left_events, right_image, right_events, n_blocks, dct_max, dct_min)
            loss = criterion(output, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:  # Print every 10 batches
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    # Step the scheduler to adjust the learning rate
    scheduler.step()
