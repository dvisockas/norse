import torch
from torch import nn
from torch.autograd import Variable
from torchaudio import transforms
import torch.nn.functional as F
import pdb

class Transformer(nn.Module):
    def __init__(self, bs=0):
        self.bs = bs
        super(Transformer, self).__init__()
        
        self.e_1 = nn.TransformerEncoderLayer(d_model=8192, nhead=1)
        #self.e_2 = nn.TransformerEncoderLayer(d_model=16384, nhead=4)
        
        #self.d_1 = nn.TransformerDecoderLayer(d_model=16384, nhead=4)
        self.d_2 = nn.TransformerDecoderLayer(d_model=8192, nhead=1)
        

    def forward(self, x):
        x = self.e_1(x)
        
        return x
