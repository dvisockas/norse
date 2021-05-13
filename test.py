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
    model.eval()
    
    for idx, file in enumerate(files):
        clean_audio, _fs = torchaudio.load(f'{TEST_DIR}/clean/{file}')
        noisy_audio, _fs = torchaudio.load(f'{TEST_DIR}/noisy/{file}')
        noisy_audio = noisy_audio.unsqueeze(0)
        
        clean_audio = clean_audio[:, :32000]
        noisy_audio = noisy_audio[:, :, :32000]
        
        signal_length = clean_audio.shape[-1]
        
        if signal_length < 32000:
            shortage = 32000 - signal_length
            clean_audio    = torch.nn.functional.pad(clean_audio, (0, shortage))
            noisy_audio    = torch.nn.functional.pad(noisy_audio, (0, shortage))
        
        with torch.no_grad():
            denoised = model(noisy_audio.cuda()).squeeze().cpu().detach().numpy()
        
        clean_for_comparison = clean_audio[0][:signal_length]
        denoised = denoised[:signal_length]

        stoi_scores.append(stoi(clean_for_comparison, denoised, fs, extended=False))
        pesq_scores.append(pesq(clean_for_comparison, denoised, fs))
        
        
    pesqs = np.array(pesq_scores)
    print(f'PESQ: {pesqs.mean()} +- {pesqs.std()}')
                                      
    stois = np.array(stoi_scores)
    print(f'STOI: {stois.mean()} +- {stois.std()}')