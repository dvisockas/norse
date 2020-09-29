import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(
        self,
        bs            = 0,
        pay_attention = True,
        encoder_type  = 'SAP',
        depth         = 6,
        channels_in   = 1,
        channels_out  = 1,
        growth_factor = 16,
        kernel_size   = 32
    ):
        self.bs = bs
        self.attn = pay_attention
        self.encoder_type = encoder_type
        self.kernel_size = kernel_size

        super(Autoencoder, self).__init__()

        padding_mode = 'reflect'

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(depth):
            encoder_ch_in = max(growth_factor * (index), 1)

            encoder_ch_out = growth_factor * (index+1)

            encode = []
            encode += [
                nn.Conv1d(
                    encoder_ch_in, encoder_ch_out, self.kernel_size, 2, 15, padding_mode=padding_mode, bias=False,
                ),
                nn.BatchNorm1d(encoder_ch_out),
                nn.GELU(),
            ]

            self.encoder.append(nn.Sequential(*encode))

            channel_multiplier = 2

            decoder_ch_in = (growth_factor * (depth-index)) * channel_multiplier
            decoder_ch_out = ((growth_factor * (depth-index-1)) * channel_multiplier) // 2

            decoder_ch_out = max(decoder_ch_out, 1)

            decode = []
            decode += [
                nn.ConvTranspose1d(
                    decoder_ch_in, decoder_ch_out, self.kernel_size, 2, 15, bias = False,
                ),
                nn.BatchNorm1d(decoder_ch_out),
            ]

            # activation = nn.Tanh() if decoder_ch_out == 1 else nn.GELU()

            activation = nn.GELU()
            if decoder_ch_out > 1: decode.append(activation)

            self.decoder.append(nn.Sequential(*decode))

        self.attention = nn.Sequential(
            nn.Conv1d(encoder_ch_out, 128, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, encoder_ch_out, kernel_size=1),
            nn.Softmax(dim=1),
            )

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        saved = [x]
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = x * w
            # x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt(
                (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5)
            )
            x = torch.cat((mu,sg), 1)

        for idx, decode in enumerate(self.decoder):
            encoder_output = saved.pop(-1)
            layer_in = torch.cat((encoder_output, x), dim=1) # Or could just enc_out + x
            x = decode(layer_in)

        return x

