import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


def get_dataset(dataset_type, path, tokenizer, max_len):
    if dataset_type == "gcdc":
        return GCDC_Dataset(path, tokenizer, max_len)
    # else if


class GCDC_Dataset(Dataset):
    def __init__(self, csv_file, tokenizer: BertTokenizer, max_len):
        data = pd.read_csv(csv_file)

        e = tokenizer.encode('hello, boy')
        self.docs = []
        self.masks = []
        for text in data['text']:
            # sentence = torch.LongTensor(tokenizer.encode(text, add_special_tokens=True))

            res = tokenizer.batch_encode_plus(text.split('\n\n'),
                                              max_length=max_len,
                                              pad_to_max_length=True,
                                              add_special_tokens=True,
                                              return_tensors='pt')
            self.docs.append(res['input_ids'])
            self.masks.append(res['attention_mask'])

        # consider only expert ratings and start as a binary classification according to the google doc
        y = data[['ratingA1', 'ratingA2', 'ratingA3']].mean(axis=1).to_numpy()
        self.y = torch.LongTensor(y >= 2)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.docs[idx], self.masks[idx], self.y[idx]


def collate_pad_fn(batch):
    x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    return x_pad, torch.LongTensor(y)
