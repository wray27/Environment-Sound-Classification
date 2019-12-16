import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode
        

    def __getitem__(self, index):
        
        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature
            LMC = np.zeros((0,41))
            for feature in self.dataset[index]['features']:
                if feature in ["logmelspec", "chroma", 'spectral_contrast', "tonnetz"]:
                    print(feature)
                    LMC = np.vstack((LMC, self.dataset[index]['features'][feature]))
            feature = torch.from_numpy(LMC.astype(np.float32)).unsqueeze(0)
        
        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MC feature
            MC = np.zeros((0,41))
            for feature in self.dataset[index]['features']:
                # print(feature)
                if feature in  ["mfcc", "chroma", 'spectral_contrast',  "tonnetz"]:
                    MC = np.vstack((MC, self.dataset[index]['features'][feature]))

         
            feature = torch.from_numpy(MC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MLMC feature
            MLMC = np.zeros((0,41))
            for feature in self.dataset[index]['features']:
                # print(feature)
                if feature in ["mfcc", "logmelspec", "chroma", 'spectral_contrast', "tonnetz"]:
                    MLMC = np.vstack((MLMC, self.dataset[index]['features'][feature]))

            
            feature = torch.from_numpy(MLMC.astype(np.float32)).unsqueeze(0)
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
