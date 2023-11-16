import torch
from torch.utils.data import Dataset
import pickle

class AmazonMusic(Dataset):
    def __init__(self, m):
      with open('texts.pkl', 'rb') as f:
        self.data = pickle.load(f)
        self.data = self.data
        self.m = m
        print(f'loading data, len: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], random.choice(range(self.m)), self.data[idx][2], self.data[idx][3]