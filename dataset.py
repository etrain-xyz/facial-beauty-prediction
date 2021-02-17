from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class FaceData(Dataset):
    def __init__(self, root, filedir, transform=None):
        self.data = []
        self.labels = []
        # Data loading
        with open(filedir, 'r') as f:
            lines = f.readlines()  

        for line in lines:
            linesplit = line.split('\n')[0].split(' ')
            addr = linesplit[0]
            target = torch.Tensor([float(linesplit[1])])
            img = Image.open(os.path.join(root, addr)).convert('RGB')

            if transform is not None:
                img = transform(img)
            self.data.append(img)
            self.labels.append(target)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
