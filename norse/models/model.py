import torch
from torch import nn
import pdb

class Autoencoder(nn.Module):
    def __init__(self, bs=0, pay_attention=True):
        self.bs = bs
        self.attn = pay_attention
        super(Autoencoder, self).__init__()

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
        self.act_11 = nn.PReLU(init=0)

        if self.attn:
            self.attn_f = nn.Conv1d(1024, 1024, 1)
            self.attn_g = nn.Conv1d(1024, 1024, 1)
            self.attn_h = nn.Conv1d(1024, 1024, 1)

        self.deconv_11 = nn.ConvTranspose1d(1024, 512, 32, 2, 15)
        self.act_d_11 = nn.PReLU(init=0)
        self.deconv_10 = nn.ConvTranspose1d(1024, 256, 32, 2, 15)
        self.act_d_10 = nn.PReLU(init=0)
        self.deconv_9 = nn.ConvTranspose1d(512, 256, 32, 2, 15)
        self.act_d_9 = nn.PReLU(init=0)
        self.deconv_8 = nn.ConvTranspose1d(512, 128, 32, 2, 15)
        self.act_d_8 = nn.PReLU(init=0)
        self.deconv_7 = nn.ConvTranspose1d(256, 128, 32, 2, 15)
        self.act_d_7 = nn.PReLU(init=0)
        self.deconv_6 = nn.ConvTranspose1d(256, 64, 32, 2, 15)
        self.act_d_6 = nn.PReLU(init=0)
        self.deconv_5 = nn.ConvTranspose1d(128, 64, 32, 2, 15)
        self.act_d_5 = nn.PReLU(init=0)
        self.deconv_4 = nn.ConvTranspose1d(128, 32, 32, 2, 15)
        self.act_d_4 = nn.PReLU(init=0)
        self.deconv_3 = nn.ConvTranspose1d(64, 32, 32, 2, 15)
        self.act_d_3 = nn.PReLU(init=0)
        self.deconv_2 = nn.ConvTranspose1d(64, 16, 32, 2, 15)
        self.act_d_2 = nn.PReLU(init=0)
        self.deconv_1 = nn.ConvTranspose1d(32, 1, 32, 2, 15)
        self.act_d_1 = nn.Tanh()

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
        c_11 = self.norm_11(self.conv_11(self.act_10(c_10)))
        c_11 = self.act_11(c_11)

        # Attention mechanism, adopted from: https://arxiv.org/pdf/1805.08318.pdf
        # Caveat: this implementation does not transpose attention function f outputs
        if self.attn:
            attn_f = self.attn_f(c_11)
            attn_g = self.attn_g(c_11)
            attn_combine = nn.Softmax(dim=1)(attn_f * attn_g)
            attn_h = self.attn_h(c_11)
            attn_out = attn_h * attn_combine

        d_11 = self.deconv_11(attn_out if self.attn else c_11)
        pre_d_10 = torch.cat((d_11, c_10), dim=1)
        d_10 = self.act_d_10(self.deconv_10(pre_d_10))
        pre_d_9 = torch.cat((d_10, c_9), dim=1)
        d_9 = self.act_d_9(self.deconv_9(pre_d_9))
        pre_d_8 = torch.cat((d_9, c_8), dim=1)
        d_8 = self.act_d_8(self.deconv_8(pre_d_8))
        pre_d_7 = torch.cat((d_8, c_7), dim=1)
        d_7 = self.act_d_7(self.deconv_7(pre_d_7))
        pre_d_6 = torch.cat((d_7, c_6), dim=1)
        d_6 = self.act_d_6(self.deconv_6(pre_d_6))
        pre_d_5 = torch.cat((d_6, c_5), dim=1)
        d_5 = self.act_d_5(self.deconv_5(pre_d_5))
        pre_d_4 = torch.cat((d_5, c_4), dim=1)
        d_4 = self.act_d_4(self.deconv_4(pre_d_4))
        pre_d_3 = torch.cat((d_4, c_3), dim=1)
        d_3 = self.act_d_3(self.deconv_3(pre_d_3))
        pre_d_2 = torch.cat((d_3, c_2), dim=1)
        d_2 = self.act_d_2(self.deconv_2(pre_d_2))
        pre_d_1 = torch.cat((d_2, c_1), dim=1)
        d_1 = self.act_d_1(self.deconv_1(pre_d_1))

        return d_1
