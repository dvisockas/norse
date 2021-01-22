from torch.utils.data.sampler import SubsetRandomSampler
from fastai.vision.all import *

class SpeechSampler():
    def __init__(self, dataset, validation, batch_size):
        dataset = dataset
        idxs = list(range(len(dataset)))
        validation = validation
        split = validation if validation > 1 else int(np.floor(validation * len(dataset)))
        train_idxs, val_idxs = idxs[split:], idxs[:split]

        train_sampler = SubsetRandomSampler(train_idxs)
        valid_sampler = SubsetRandomSampler(val_idxs)

        self.train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=True, drop_last=True)
        self.val_dl = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=True, drop_last=True)
        
    def __call__(self):
        return (self.train_dl, self.val_dl)