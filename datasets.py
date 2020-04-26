import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class GCDC_Dataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        data = pd.read_csv(csv_file)
        self.x = [torch.LongTensor(tokenizer.encode(text, add_special_tokens=True)) for text in data['text']]
        # consider only expert ratings and start as a binary classification according to the google doc
        y = data[['ratingA1', 'ratingA2', 'ratingA3']].mean(axis=1).to_numpy()
        self.y = torch.LongTensor(y > 2.5)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def collate_pad_fn(batch):
    x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    return x_pad, torch.LongTensor(y)
