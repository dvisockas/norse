import torch
from torch import nn
from torch.autograd import Variable
from torchaudio import transforms
import torch.nn.functional as F
import pdb
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.in_dropout = nn.Dropout(p=0.3)
        
        self.conv_1 = nn.Conv1d(1, 16, 32, 2, 15)
        self.norm_1 = nn.PReLU()
        self.conv_2 = nn.Conv1d(16, 32, 32, 2, 15)
        self.norm_2 = nn.PReLU()
        self.conv_3 = nn.Conv1d(32, 32, 32, 2, 15)
        self.norm_3 = nn.PReLU()
        self.conv_4 = nn.Conv1d(32, 64, 32, 2, 15)
        self.norm_4 = nn.PReLU()
        self.conv_5 = nn.Conv1d(64, 64, 32, 2, 15)
        self.norm_5 = nn.PReLU()
        self.conv_6 = nn.Conv1d(64, 128, 32, 2, 15)
        self.norm_6 = nn.PReLU()
        self.conv_7 = nn.Conv1d(128, 128, 32, 2, 15)
        self.norm_7 = nn.PReLU()
        self.conv_8 = nn.Conv1d(128, 256, 32, 2, 15)
        self.norm_8 = nn.PReLU()
        self.conv_9 = nn.Conv1d(256, 256, 32, 2, 15)
        self.norm_9 = nn.PReLU()
        self.conv_10 = nn.Conv1d(256, 512, 32, 2, 15)
        self.norm_10 = nn.PReLU()
        self.conv_11 = nn.Conv1d(512, 1024, 32, 2, 15)
        self.norm_11 = nn.PReLU()
        
        self.deconv_11 = nn.ConvTranspose1d(1024, 512, 32, 2, 15)
        self.norm_d_11 = nn.PReLU()
        self.deconv_10 = nn.ConvTranspose1d(1024, 256, 32, 2, 15)
        self.norm_d_10 = nn.PReLU()
        self.deconv_9 = nn.ConvTranspose1d(512, 256, 32, 2, 15)
        self.norm_d_9 = nn.PReLU()
        self.deconv_8 = nn.ConvTranspose1d(512, 128, 32, 2, 15)
        self.norm_d_8 = nn.PReLU()
        self.deconv_7 = nn.ConvTranspose1d(256, 128, 32, 2, 15)
        self.norm_d_7 = nn.PReLU()
        self.deconv_6 = nn.ConvTranspose1d(256, 64, 32, 2, 15)
        self.norm_d_6 = nn.PReLU()
        self.deconv_5 = nn.ConvTranspose1d(128, 64, 32, 2, 15)
        self.norm_d_5 = nn.PReLU()
        self.deconv_4 = nn.ConvTranspose1d(128, 32, 32, 2, 15)
        self.norm_d_4 = nn.PReLU()
        self.deconv_3 = nn.ConvTranspose1d(64, 32, 32, 2, 15)
        self.norm_d_3 = nn.PReLU()
        self.deconv_2 = nn.ConvTranspose1d(64, 16, 32, 2, 15)
        self.norm_d_2 = nn.PReLU()
        self.deconv_1 = nn.ConvTranspose1d(32, 1, 32, 2, 15)
        self.norm_d_1 = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.in_dropout(x)
        c_1 = self.conv_1(x)
        c_2 = self.conv_2(self.norm_1(c_1))
        c_3 = self.conv_3(self.norm_2(c_2))
        c_4 = self.conv_4(self.norm_3(c_3))
        c_5 = self.conv_5(self.norm_4(c_4))
        c_6 = self.conv_6(self.norm_5(c_5))
        c_7 = self.conv_7(self.norm_6(c_6))
        c_8 = self.conv_8(self.norm_7(c_7))
        c_9 = self.conv_9(self.norm_8(c_8))
        c_10 = self.conv_10(self.norm_9(c_9))
        c_11 = self.conv_11(self.norm_10(c_10))
        c_11 = self.norm_11(c_11)

        d_11 = self.deconv_11(c_11)
        pre_d_10 = torch.cat((d_11, c_10), dim=1)
        d_10 = self.norm_d_10(self.deconv_10(pre_d_10))
        pre_d_9 = torch.cat((d_10, c_9), dim=1)
        d_9 = self.norm_d_9(self.deconv_9(pre_d_9))
        pre_d_8 = torch.cat((d_9, c_8), dim=1)
        d_8 = self.norm_d_8(self.deconv_8(pre_d_8))
        pre_d_7 = torch.cat((d_8, c_7), dim=1)
        d_7 = self.norm_d_7(self.deconv_7(pre_d_7))
        pre_d_6 = torch.cat((d_7, c_6), dim=1)
        d_6 = self.norm_d_6(self.deconv_6(pre_d_6))
        pre_d_5 = torch.cat((d_6, c_5), dim=1)
        d_5 = self.norm_d_5(self.deconv_5(pre_d_5))
        pre_d_4 = torch.cat((d_5, c_4), dim=1)
        d_4 = self.norm_d_4(self.deconv_4(pre_d_4))
        pre_d_3 = torch.cat((d_4, c_3), dim=1)
        d_3 = self.norm_d_3(self.deconv_3(pre_d_3))
        pre_d_2 = torch.cat((d_3, c_2), dim=1)
        d_2 = self.norm_d_2(self.deconv_2(pre_d_2))
        pre_d_1 = torch.cat((d_2, c_1), dim=1)
        d_1 = self.norm_d_1(self.deconv_1(pre_d_1))

        return d_1.reshape(128, 16384)
