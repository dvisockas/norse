import torch
from pypesq import pesq
from pystoi import stoi
import os
import torchaudio
import numpy as np

TEST_DIR = '/home/dan/www/norse/data/test/'

def test(model):
    fs = 16000
    files = os.listdir(f'{TEST_DIR}clean')
    pesq_scores = []
    stoi_scores = []
    
    for idx, file in enumerate(files):
        clean_audio, _fs = torchaudio.load(f'{TEST_DIR}/clean/{file}')
        noisy_audio, _fs = torchaudio.load(f'{TEST_DIR}/noisy/{file}')
        noisy_audio = noisy_audio.unsqueeze(0)
                                      
        with torch.no_grad():
            denoised = model(noisy_audio).squeeze().cpu().detach().numpy()
        
        denoised = np.pad(denoised, [0,1])
        stoi_scores.append(stoi(clean_audio[0], denoised, fs, extended=False))
        pesq_scores.append(pesq(clean_audio[0], denoised, fs))
        
        
    pesqs = np.array(pesq_scores)
    print(f'PESQ: {pesqs.mean()} +- {pesqs.std()}')
                                      
    stois = np.array(stoi_scores)
    print(f'STOI: {stois.mean()} +- {stois.std()}')