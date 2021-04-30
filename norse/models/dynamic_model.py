import torch
from torch import nn

UPSAMPLE_TYPES  = ['transpose', 'conv']
ATTENTION_TYPES = ['SAP', 'ASP']
SKIP_OP_TYPES   = ['add', 'cat']

class Autoencoder(nn.Module):
    def __init__(
        self,
        pay_attention = True,
        encoder_type  = 'SAP',
        depth         = 6,
        channels_in   = 1,
        channels_out  = 1,
        growth_factor = 16,
        kernel_size   = 16,
        upsample_type = 'transpose',
        skip_op       = 'add'
    ):
        self.attn = pay_attention
        self.depth = depth
        self.growth_factor = growth_factor
        self.encoder_type = encoder_type
        assert(encoder_type in ATTENTION_TYPES)

        self.kernel_size = kernel_size
        self.upsample_type = upsample_type
        if pay_attention:
            assert(upsample_type in UPSAMPLE_TYPES)

        self.padding_mode = 'reflect'
        
        assert(skip_op in SKIP_OP_TYPES)
        self.skip_op = skip_op

        super(Autoencoder, self).__init__()
        self.bootstrap()

    def bootstrap(self):
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(self.depth):
            encoder_ch_in = max(self.growth_factor * (index), 1)
            encoder_ch_out = self.growth_factor * (index + 1)

            encode = []
            encode += [
                nn.Conv1d(
                    encoder_ch_in, encoder_ch_out, self.kernel_size, stride=2, padding=7, padding_mode=self.padding_mode, bias=False,
                ),
                nn.BatchNorm1d(encoder_ch_out),
                nn.GELU(),
            ]

            self.encoder.append(nn.Sequential(*encode))

            channel_multiplier = 2 if self.skip_op == 'cat' else 1
            
            growth = self.growth_factor * channel_multiplier
            decoder_ch_in = growth * (self.depth-index)
            decoder_ch_out = (growth * (self.depth-index-1)) #// 2
            decoder_ch_out = max(decoder_ch_out, 1)

            stride = self.kernel_size if decoder_ch_out == 0 else 1
            # TODO: Ensure that stride is a mod of kernel size and matches in skip connections
            stride = 2

            decode = []
            if self.upsample_type == 'transpose':
                decode += [
                    nn.ConvTranspose1d(
                        decoder_ch_in, decoder_ch_out, self.kernel_size, stride=stride, padding=7, bias=False,
                    )
                ]
            elif self.upsample_type == 'conv':
                decode += [
                    nn.Upsample(decoder_ch_out),
                    nn.Conv1d(decoder_ch_out, decoder_ch_out, self.kernel_size, stride=stride, padding=7, bias=False)
                ]

            activation = nn.Tanh() if decoder_ch_out == 1 else nn.GELU()

            #activation = nn.GELU()
            if decoder_ch_out > 1:
                decode.append(nn.BatchNorm1d(decoder_ch_out))
                decode.append(activation)

            self.decoder.append(nn.Sequential(*decode))


        self.attention = nn.Sequential(
            nn.Conv1d(encoder_ch_out, 128, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, encoder_ch_out, kernel_size=1),
            nn.Softmax(dim=1),
            )

        self.resolver = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=True),
            nn.Tanh()
        )
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def attend(self, x):
        w = self.attention(x)

        if self.encoder_type == 'SAP':
            x = x * w
            # x = torch.sum(x * w, dim=2)
        elif self.encoder_type == 'ASP':
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt(
                (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5)
            )
            x = torch.cat((mu,sg), 1)

        return x
    
    def resolve_skip_op(self, skip_in, another_skip_in):
        print(skip_in.shape)
        print(another_skip_in.shape)
        if self.skip_op == 'add':
            return skip_in + another_skip_in
        elif self.skip_op == 'cat':
            return torch.cat((skip_in, another_skip_in), dim=1)

    def forward(self, x):
        saved = [x]
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        x = self.attend(x)

        for decode in self.decoder:
            encoder_output = saved.pop(-1)
            layer_in = self.resolve_skip_op(encoder_output, x)
            #layer_in = torch.cat((encoder_output, x), dim=1)
            x = decode(layer_in)

        x = self.resolver(x)

        return x

