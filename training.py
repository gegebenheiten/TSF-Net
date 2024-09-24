import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.EDSR import EDSR
from dataset import demoDataset
import numpy  as np
import os
from basic_potion import SimpleOptions

option = SimpleOptions()
opt = option.parse()
opt.isTrain = True 
# Prepare data
train_dataset = demoDataset(opt)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

# Initialize model, criterion, and optimizer
model = EDSR()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

# Training loop
for epoch in range(opt.num_epochs):
    for se_idx in range(len(opt.senarios)):
        opt.se_idx = se_idx
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            #for t in range():
            # Zero the parameter gradients
            optimizer.zero_grad() 

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:  # Print every 10 batches
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    # Step the scheduler to adjust the learning rate
    scheduler.step()
