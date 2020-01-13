import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import pickle


def show_graph( name, data):

    plt.imshow(data, origin='lower')
    plt.xlabel("Frame")
    plt.ylabel("Frequency (Hz)")
    cbar = plt.colorbar()
    cbar.set_label("Volume (dB)")
    plt.title(f"{name} Feature Set")
    
    plt.savefig(name + ".png", bbox_inches='tight')
    plt.show()

class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode
        

    def __getitem__(self, index):
        # Possibly might need to fix this
        # keys don't print in the right order
        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature
            LMC = np.zeros((0,41))
            for feature in self.dataset[index]['features']:
                # print(feature)
                LMC = np.vstack((LMC, self.dataset[index]['features']["logmelspec"]))
                LMC = np.vstack((LMC, self.dataset[index]['features']["chroma"]))
                LMC = np.vstack((LMC, self.dataset[index]['features']["spectral_contrast"))
                LMC = np.vstack((LMC, self.dataset[index]['features']["tonnetz"]))

            feature = torch.from_numpy(LMC.astype(np.float32)).unsqueeze(0)

            if index == 12:
                show_graph("LMC", LMC)

        
        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MC feature
            MC = np.zeros((0,41))
            for feature in self.dataset[index]['features']:
                # print(feature)
                if feature in  ["mfcc", "chroma", 'spectral_contrast',  "tonnetz"]:
                    MC = np.vstack((MC, self.dataset[index]['features'][feature]))
                
            if index == 12:
                show_graph("MC",MC)

         
            feature = torch.from_numpy(MC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MLMC feature
            MLMC = np.zeros((0,41))
            for feature in self.dataset[index]['features']:
                # print(feature)
                # if feature in ["mfcc", "logmelspec", "chroma", 'spectral_contrast', "tonnetz"]:
                MLMC = np.vstack((MLMC, self.dataset[index]['features']["mfcc"]))
                MLMC = np.vstack((MLMC, self.dataset[index]['features']["logmelspec"]))
                MLMC = np.vstack((MLMC, self.dataset[index]['features']["chroma"]))
                MLMC = np.vstack((MLMC, self.dataset[index]['features']['spectral_contrast']))
                MLMC = np.vstack((MLMC, self.dataset[index]['features']["tonnetz"]))
                

            
            feature = torch.from_numpy(MLMC.astype(np.float32)).unsqueeze(0)
        
        # feature = torch.flatten(feature,0,1)
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        # print(feature.shape)

        return feature, label, fname


    def __len__(self):
        return len(self.dataset)

def main():
    
    x = UrbanSound8KDataset('UrbanSound8K_test.pkl', "LMC").__getitem__(12)
    y = UrbanSound8KDataset('UrbanSound8K_test.pkl', "MC").__getitem__(12)
    

if __name__ == '__main__':
    main()
    