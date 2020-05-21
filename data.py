import torch
import torchaudio
from pathlib import Path
import numpy as np
import glob
from torch.utils.data.dataset import Dataset
import pdb
from utils import windows
import json
import os

FILE_COUNT_CACHE = 'file-count-cache.json'

class CleanSpeechDataset(Dataset):
    def __init__(self, data_dir, window_size, overlap):
        SAMPLE_RATE = 16_000
        
        self.window_size = window_size# int(SAMPLE_RATE * (window_size / 1000))
        self.overlap = int(self.window_size * (overlap / 100))
        self.sound_files_list = glob.glob(data_dir + '*')
        self.data_len = 0
        self.sound_files_by_length = dict()
        
        if os.path.exists(FILE_COUNT_CACHE):
            with open(FILE_COUNT_CACHE) as json_file:
                self.sound_files_by_length = json.load(json_file)
            self.data_len = sum([length for filename, length in self.sound_files_by_length.items()])
        else:
            for filename in self.sound_files_list:
                wave, _sample_rate = torchaudio.load(filename)
                sample_length = len(wave.T) // self.window_size
                self.data_len += sample_length
                self.sound_files_by_length[filename] = sample_length
            
            file_lengths = json.dumps(self.sound_files_by_length)
            f = open(FILE_COUNT_CACHE, 'w')
            f.write(file_lengths)
            f.close()
                
        self.data_len = int(self.data_len) - 1
    
    def find_filename(self, index):
        counter = -1
        for filename, count in self.sound_files_by_length.items(): 
            counter += count
            if index < counter:
                return (filename, index - (counter - count)) 

    def __getitem__(self, index):
        filename, nth_sample = self.find_filename(index)
        wave, _sample_rate = torchaudio.load(filename)
        sample = windows(wave, wsize=self.window_size)[:, nth_sample]
        return [sample, sample]

    def __len__(self):
        return self.data_len
