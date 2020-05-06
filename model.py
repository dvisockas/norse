import torch
from torch import nn
from torch.autograd import Variable
from torchaudio import transforms
import torch.nn.functional as F
import pdb

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        FILTER_SIZE = 31
        
        self.in_dropout = nn.Dropout(p=0.3)
        
        self.conv1 = nn.Conv1d(1, 16, FILTER_SIZE, stride=2, padding=15)
        self.conv2 = nn.Conv1d(16, 32, FILTER_SIZE, stride=2, padding=15)
        self.conv3 = nn.Conv1d(32, 32, FILTER_SIZE, stride=2, padding=15)
        self.conv4 = nn.Conv1d(32, 64, FILTER_SIZE, stride=2, padding=15)
        self.conv5 = nn.Conv1d(64, 64, FILTER_SIZE, stride=2, padding=15)
        self.conv6 = nn.Conv1d(64, 128, FILTER_SIZE, stride=2, padding=15)
        self.conv7 = nn.Conv1d(128, 128, FILTER_SIZE, stride=2, padding=15)
        self.conv8 = nn.Conv1d(128, 256, FILTER_SIZE, stride=2, padding=15)
        self.conv9 = nn.Conv1d(256, 256, FILTER_SIZE, stride=2, padding=15)
        self.conv10 = nn.Conv1d(256, 512, FILTER_SIZE, stride=2, padding=15)
        self.conv11 = nn.Conv1d(512, 1024, FILTER_SIZE, stride=2, padding=15)
        
        self.deconv1 = nn.ConvTranspose1d(1024, 512, FILTER_SIZE, stride=3, padding=18)
        self.deconv2 = nn.ConvTranspose1d(512, 256, FILTER_SIZE, stride=2, padding=15)
        self.deconv3 = nn.ConvTranspose1d(256, 256, FILTER_SIZE, stride=2, padding=15)
        self.deconv4 = nn.ConvTranspose1d(256, 128, FILTER_SIZE, stride=2, padding=15)
        self.deconv5 = nn.ConvTranspose1d(128, 128, FILTER_SIZE, stride=2, padding=15)
        self.deconv6 = nn.ConvTranspose1d(128, 64, FILTER_SIZE, stride=2, padding=15)
        self.deconv7 = nn.ConvTranspose1d(64, 64, FILTER_SIZE, stride=2, padding=15)
        self.deconv8 = nn.ConvTranspose1d(64, 32, FILTER_SIZE, stride=2, padding=15)
        self.deconv9 = nn.ConvTranspose1d(32, 32, FILTER_SIZE, stride=2, padding=15)
        self.deconv10 = nn.ConvTranspose1d(32, 16, FILTER_SIZE, stride=2, padding=15)
        self.deconv11 = nn.ConvTranspose1d(16, 1, FILTER_SIZE, stride=2, padding=15)

    def forward(self, x):
        x = self.in_dropout(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        c10 = F.relu(x)
        x = self.conv11(c10)
        x = F.relu(x)

        x = self.deconv1(x)
        x = F.relu(x)
        pdb.set_trace()
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.relu(x)
        x = self.deconv4(x)
        x = F.relu(x)
        x = self.deconv5(x)
        x = F.relu(x)
        x = self.deconv6(x)
        x = F.relu(x)
        x = self.deconv7(x)
        x = F.relu(x)
        x = self.deconv8(x)
        x = F.relu(x)
        x = self.deconv9(x)
        x = F.relu(x)
        x = self.deconv10(x)
        x = F.relu(x)
        x = self.deconv11(x)
        x = F.relu(x)

        return x