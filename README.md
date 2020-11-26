# Norse - Non Recurrent Speech Enhancement
My master thesis project exploring various ways to enchance speech signals in a fully-convolutional fashion.

Current features:
- Variable depth encoder/decoder
- Modular attention module (currently only SAP and ASP are implemented)
- Variable upsampling (Upsample+Conv or DeConv)

### Introduction
The work is inspired by SEGAN and DEMUCS architectures. Both of them follow a similar U-net-like design:

Input -> Encoder -> Bottleneck ops (RNN, Noise sampling, w/e) -> Decoder -> Output <--> Loss func (D in case of SEGAN, MSE in demucs)

The idea is to explore what are the best bottleneck operations, loss funcs, optimal depth, upsampling methods and so on.
