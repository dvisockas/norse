import torch
from torch import nn
import pdb

class Wavenetish(nn.Module):
    def __init__(self, bs=0, pay_attention=True):
        self.bs = bs
        self.attn = pay_attention
        super(Wavenetish, self).__init__()

        padding_mode = 'reflect'

        self.conv_1 = nn.Conv1d(1, 16, 32, 2, 15, padding_mode=padding_mode)
        self.norm_1 = nn.BatchNorm1d(16)
        self.act_1 = nn.PReLU(init=0)

        self.conv_2 = nn.Conv1d(16, 32, 32, 2, 15, padding_mode=padding_mode)
        self.norm_2 = nn.BatchNorm1d(32)
        self.act_2 = nn.PReLU(init=0)

        self.conv_3 = nn.Conv1d(32, 32, 32, 2, 15, padding_mode=padding_mode)
        self.norm_3 = nn.BatchNorm1d(32)
        self.act_3 = nn.PReLU(init=0)

        self.conv_4 = nn.Conv1d(32, 64, 32, 2, 15, padding_mode=padding_mode)
        self.norm_4 = nn.BatchNorm1d(64)
        self.act_4 = nn.PReLU(init=0)

        self.conv_5 = nn.Conv1d(64, 64, 32, 2, 15, padding_mode=padding_mode)
        self.norm_5 = nn.BatchNorm1d(64)
        self.act_5 = nn.PReLU(init=0)

        self.conv_6 = nn.Conv1d(64, 128, 32, 2, 15, padding_mode=padding_mode)
        self.norm_6 = nn.BatchNorm1d(128)
        self.act_6 = nn.PReLU(init=0)

        self.conv_7 = nn.Conv1d(128, 128, 32, 2, 15, padding_mode=padding_mode)
        self.norm_7 = nn.BatchNorm1d(128)
        self.act_7 = nn.PReLU(init=0)

        self.conv_8 = nn.Conv1d(128, 256, 32, 2, 15, padding_mode=padding_mode)
        self.norm_8 = nn.BatchNorm1d(256)
        self.act_8 = nn.PReLU(init=0)

        self.conv_9 = nn.Conv1d(256, 256, 32, 2, 15, padding_mode=padding_mode)
        self.norm_9 = nn.BatchNorm1d(256)
        self.act_9 = nn.PReLU(init=0)

        self.conv_10 = nn.Conv1d(256, 512, 32, 2, 15, padding_mode=padding_mode)
        self.norm_10 = nn.BatchNorm1d(512)
        self.act_10 = nn.PReLU(init=0)

        self.conv_11 = nn.Conv1d(512, 1024, 32, 2, 15, padding_mode=padding_mode)
        self.norm_11 = nn.BatchNorm1d(1024)

        self.act_11 = nn.Tanh()

        if self.attn:
            self.attn_f = nn.Conv1d(512, 1024, 1)
            self.attn_g = nn.Conv1d(512, 1024, 1)
            self.attn_h = nn.Conv1d(512, 1024, 1)

        self.fc_1 = nn.Linear(8192, 4096)
        self.fc_2 = nn.Linear(4096, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        c_1 = self.norm_1(self.conv_1(x))
        c_2 = self.norm_2(self.conv_2(self.act_1(c_1)))
        c_3 = self.norm_3(self.conv_3(self.act_2(c_2)))
        c_4 = self.norm_4(self.conv_4(self.act_3(c_3)))
        c_5 = self.norm_5(self.conv_5(self.act_4(c_4)))
        c_6 = self.norm_6(self.conv_6(self.act_5(c_5)))
        c_7 = self.norm_7(self.conv_7(self.act_6(c_6)))
        c_8 = self.norm_8(self.conv_8(self.act_7(c_7)))
        c_9 = self.norm_9(self.conv_9(self.act_8(c_8)))
        c_10 = self.norm_10(self.conv_10(self.act_9(c_9)))

        if self.attn:
            attn_f = self.attn_f(c_10)
            attn_g = self.attn_g(c_10)
            attn_combine = torch.nn.Softmax(dim=1)(attn_f * attn_g)
            attn_h = self.attn_h(c_10)
            attn_out = attn_h * attn_combine

            c_10 = attn_out

        fc_1 = self.fc_1(torch.flatten(c_10, start_dim=1))
        fc_2 = self.fc_2(fc_1)

        return fc_2
