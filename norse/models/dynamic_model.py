import torch
from torch import nn

UPSAMPLE_TYPES  = ['transpose', 'conv']
ATTENTION_TYPES = ['SAP', 'ASP']
SKIP_OP_TYPES   = ['add', 'cat', 'none']
NORM_TYPES      = ['batch', 'layer']

class Autoencoder(nn.Module):
    def __init__(
        self,
        pay_attention = 0,
        depth         = 6,
        channels_in   = 1,
        channels_out  = 1,
        growth_factor = 16,
        kernel_size   = 16,
        upsample_type = 'transpose',
        skip_op       = 'add',
        norm          = 'batch'
    ):
        self.attn = pay_attention
        self.depth = depth
        self.growth_factor = growth_factor

        self.kernel_size = kernel_size
        self.norm = norm
        assert(norm in NORM_TYPES)
        
        self.upsample_type = upsample_type
        assert(upsample_type in UPSAMPLE_TYPES)

        self.padding_mode = 'reflect'
        
        assert(skip_op in SKIP_OP_TYPES)
        self.skip_op = skip_op

        super(Autoencoder, self).__init__()
        self.bootstrap()

    def bootstrap(self):
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()

        for index in range(self.depth):
            encoder_ch_in = max(self.growth_factor * (index), 1)
            encoder_ch_out = self.growth_factor * (index + 1)
            norm = nn.BatchNorm1d if self.norm == 'batch' else nn.LayerNorm
            
            encode = []
            encode += [
                nn.Conv1d(
                    encoder_ch_in, encoder_ch_out, self.kernel_size, stride=2, padding=15, padding_mode=self.padding_mode, bias=False,
                ),
                norm(encoder_ch_out),
                nn.GELU(),
            ]

            self.encoder.append(nn.Sequential(*encode))

            channel_multiplier = 2 if (self.skip_op == 'cat' and index != 0) else 1
            
            growth = self.growth_factor * channel_multiplier
            
            decoder_ch_in = growth * (self.depth-index)
            decoder_ch_out = (growth * (self.depth-index-1))
            if self.skip_op == 'cat' and index != 0: decoder_ch_out = decoder_ch_out // 2
            decoder_ch_out = max(decoder_ch_out, 1)

            stride = self.kernel_size if decoder_ch_out == 0 else 1
            # TODO: Ensure that stride is a mod of kernel size and matches in skip connections
            stride = 2

            decode = []
            if self.upsample_type == 'transpose':
                decode += [
                    nn.ConvTranspose1d(
                        decoder_ch_in, decoder_ch_out, self.kernel_size, stride=stride, padding=15, bias=False,
                    )
                ]
            elif self.upsample_type == 'conv':
                decode += [
                    nn.Upsample(decoder_ch_out),
                    nn.Conv1d(decoder_ch_out, decoder_ch_out, self.kernel_size, stride=stride, padding=7, bias=False)
                ]

            activation = nn.Tanh() if decoder_ch_out == 1 else nn.GELU()

            if decoder_ch_out > 1:
                decode.append(norm(decoder_ch_out))
                decode.append(activation)

            self.decoder.append(nn.Sequential(*decode))
        
        attn_dims = self.growth_factor * self.depth
        
        for i in range(self.attn):
            one_filter = [
                nn.Conv1d(attn_dims, attn_dims, 1),
                nn.Conv1d(attn_dims, attn_dims, 1),
                nn.Conv1d(attn_dims, attn_dims, 1)
            ]
            
            self.attention.append(nn.Sequential(*one_filter))
        if self.attn > 1:
            self.attention_resolver = nn.Conv1d(attn_dims * self.attn, attn_dims, 1, 1)

        self.resolver = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def attend(self, x):
        attention_states = []
        for attention_block in self.attention:
            f = attention_block[0](x)
            g = attention_block[1](x)
            h = attention_block[2](x)
            combine = nn.Softmax(dim=1)(f * g)
            attention_states.append(h * combine)
        attention_states = torch.cat(attention_states, dim=1)
        
        if self.attn > 1:
            return self.attention_resolver(attention_states)
        else:
            return attention_states[0]
    
    def resolve_skip_op(self, skip_in, another_skip_in):
        if self.skip_op == 'add':
            return skip_in + another_skip_in
        elif self.skip_op == 'cat':
            return torch.cat((skip_in, another_skip_in), dim=1)
        elif self.skip_op == 'none':
            return another_skip_in

    def forward(self, x):
        saved = [x]
        
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
            
        if self.attn > 0:
            x = self.attend(x)
            
        for index, decode in enumerate(self.decoder):
            encoder_output = saved.pop(-1)
            layer_in = encoder_output
            if index != 0 and self.skip_op == 'cat':
                layer_in = self.resolve_skip_op(encoder_output, x)
            x = decode(layer_in)
            

        x = self.resolver(x)

        return x

