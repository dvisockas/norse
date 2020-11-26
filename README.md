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

### Motivation
I really liked the design of SEGAN, but GANs are rather unstable and deconv layers lead to checkerboard patterns (buzzing in 1D space).

DEMUCS performs really well, but they use a LSTM module as their bottleneck operation, which we all know is slow to train.
