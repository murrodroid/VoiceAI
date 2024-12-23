import torch
from torch.utils.data import Dataset

class VoiceDataset(Dataset):
    def __init__(self, spectrograms, transcripts):
        self.spectrograms = spectrograms
        self.transcripts = transcripts

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.transcripts[idx]