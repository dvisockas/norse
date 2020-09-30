import torch
import torchaudio
from torchaudio.transforms import *
from pathlib import Path
import numpy as np
import glob
from torch.utils.data.dataset import Dataset
import pdb
from utils import windows
import json
import os
import math

torchaudio.set_audio_backend('sox_io')

FILE_COUNT_CACHE = './data/.file-count-cache.json'
NOISE_CACHE      = './data/.noise-count-cache.json'

class SpeechDataset(Dataset):
    def __init__(
        self,
        clean_dir,
        noise_dir,
        window_size,
        overlap,
        snr = 0,
        limit_samples = 0,
        output_one=False,
        window_step=None,
    ):
        self.window_step = window_step
        self.window_size = window_size
        self.overlap = overlap
        self.sound_files_list = glob.glob(clean_dir + '*')
        self.noise_files_list = glob.glob(noise_dir + '*')
        self.data_len = 0
        self.noise_len = 0
        self.sound_files_by_length = dict()
        self.noise_files_by_length = dict()
        self.limit_samples = limit_samples
        self.snr = snr
        self.output_one = output_one

        if os.path.exists(FILE_COUNT_CACHE):
            with open(FILE_COUNT_CACHE) as json_file:
                self.sound_files_by_length = json.load(json_file)
            self.data_len = sum([length // self.window_size for filename, length in self.sound_files_by_length.items()])
        else:
            for filename in self.sound_files_list:
                wave, _sample_rate = torchaudio.load(filename)
                sample_length = len(wave.T)
                self.data_len += sample_length
                self.sound_files_by_length[filename] = sample_length

            file_lengths = json.dumps(self.sound_files_by_length)
            f = open(FILE_COUNT_CACHE, 'w')
            f.write(file_lengths)
            f.close()

        if os.path.exists(NOISE_CACHE):
            with open(NOISE_CACHE) as json_file:
                self.noise_files_by_length = json.load(json_file)
            self.noise_len = sum([length // self.window_size for filename, length in self.noise_files_by_length.items()])
        else:
            for filename in self.noise_files_list:
                wave, _sample_rate = torchaudio.load(filename)
                sample_length = len(wave.T)
                self.noise_len += sample_length
                self.noise_files_by_length[filename] = sample_length

            file_lengths = json.dumps(self.noise_files_by_length)
            f = open(NOISE_CACHE, 'w')
            f.write(file_lengths)
            f.close()
        self.noise_file_names = list(self.noise_files_by_length.keys())

    def normalize(self, sample):
        # return (sample-sample.mean()) / sample.std()
        return sample

    def find_filename(self, index):
        counter = -1

        for filename, count in self.sound_files_by_length.items():
            windows_count = count // self.window_size
            counter += windows_count

            if index < counter:
                return (filename, index - (counter - windows_count))

    def find_noisefile(self, index):
        return self.noise_file_names[int((index / self.data_len) * len(self.noise_file_names))]

    def __getitem__(self, index):
        clean_file, nth_sample = self.find_filename(index)
        noise_file = self.find_noisefile(index)
        noise_wave = self.normalize(torchaudio.load(noise_file)[0])
        clean_wave = self.normalize(torchaudio.load(clean_file)[0])

        noise_len = len(noise_wave[0, :])
        clean_len = len(clean_wave[0, :])

        if noise_len < clean_len:
            repeat_times = math.ceil(clean_len / noise_len)
            noise_wave = noise_wave.repeat((1, repeat_times))

        noised_wave = torch.add(clean_wave[0, :], noise_wave[0, :clean_len] / self.snr).reshape(1, -1)

        window_args = { "window_size": self.window_size, "overlap": self.overlap }#, "step": self.window_step }

        clean_sample = windows(clean_wave, **window_args)[:, nth_sample]
        noised_sample = windows(noised_wave, **window_args)[:, nth_sample]

        if self.output_one: clean_sample = MuLawEncoding()(clean_sample[:, -1])

        return [noised_sample, clean_sample]

    def __len__(self):
        return self.limit_samples if self.limit_samples > 0 else (self.data_len - 1)
