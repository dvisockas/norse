import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchaudio import transforms
from data import SpeechDataset
import time
from model import Autoencoder
import pdb
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torchaudio import transforms
import torch.nn.functional as F
import torchaudio

num_epochs = 50
batch_size = 1024
learning_rate = 1e-3

dataset = SpeechDataset('data/clean/open_slr/360', 'data/clean/noise/', 16384, 50)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model.train()
for epoch in range(num_epochs):
    start = time.time()

    for i, data in enumerate(dataloader):
        sample = data
        sample = Variable(sample).cuda()
        output = model(sample)
        loss = criterion(output, sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch == 0:
        time_diff = time.time() - start
        print(f'Epoch took {round(time_diff, 4)}s')
        print(f'Expected hours to train: {time_diff * num_epochs / 3600 }')
    if epoch % 5 == 0:
        print(f'epoch [{epoch+1}/{num_epochs}]')
        print(round(loss.item(), 5))