import torch
from torch.utils import data
import numpy as np
import pickle
import librosa

class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode
        print(len(self.dataset))

    def __getitem__(self, index):
        features = []

        if self.mode == 'LMC':
            log_mel_spec = self.dataset[index]['features']['logmelspec']
            features.append(torch.from_numpy(log_mel_spec.astype(np.float32)).unsqueeze(0))
            

        elif self.mode == 'MC':
            mfcc     = self.dataset[index]['features']['mfcc']
            features.append(torch.from_numpy(mfcc.astype(np.float32)).unsqueeze(0))
            
            
        chroma            = self.dataset[index]['features']['chroma']
        features.append(torch.from_numpy(chroma.astype(np.float32)).unsqueeze(0))
    
        spectral_contrast = self.dataset[index]['features']['spectral_contrast']
        features.append(torch.from_numpy(spectral_contrast.astype(np.float32)).unsqueeze(0))
        
        tonnetz           = self.dataset[index]['features']['tonnetz']
        features.append(torch.from_numpy(tonnetz.astype(np.float32)).unsqueeze(0))

        features = torch.cat(features, dim=1)
        

        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return features, label, fname

    def __len__(self):
        return len(self.dataset)