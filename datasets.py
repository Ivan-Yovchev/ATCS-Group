import os
import pandas as pd

from torch import LongTensor, stack, float32 as tfloat32
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from transformers import BertTokenizer

from random import shuffle
from math import ceil

def get_dataset(dataset_type, path, tokenizer, max_len, batch_size, device):
    if dataset_type == "gcdc":
        return GCDC_Dataset(path, tokenizer, max_len)
    elif dataset_type == "hyperpartisan":
        return HyperpartisanDataset(path, tokenizer, max_len)
    # else if
    return None

class GCDC_Dataset(Dataset):
    def __init__(self, csv_file, tokenizer: BertTokenizer, max_len, batch_size, device):

        self.batch_size = batch_size
        self.device = device

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
        self.y = LongTensor(y >= 2)

        self.__shuffle()

    def __shuffle(self):

        # Shuffle docs
        temp = list(zip(self.docs, self.masks, self.y))
        shuffle(temp)

        self.docs, self.masks, self.y = zip(*temp)

        self.idx = 0

    def __len__(self):
        return ceil(len(self.y)/self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):

        if self.idx >= len(self):
            self.__shuffle()
            raise StopIteration

        # Get interval of current batch
        idx_start, idx_end = self.batch_size*self.idx, min(self.batch_size*(self.idx+1), len(self.y))

        # Get batch lsits
        docs, masks, ys = self.docs[idx_start:idx_end], self.masks[idx_start:idx_end], self.y[idx_start:idx_end]

        # Pad docs (to have the same number of sentences)
        lengths = [doc.shape[0] for doc in docs]
        max_sent = max(lengths)

        docs = stack(
            [
                F.pad(doc, pad=(0,0,0,max_sent-doc.shape[0]))
                for doc in docs
            ]
        ).to(self.device)

        masks = stack(
            [
                F.pad(mask, pad=(0,0,0,max_sent-mask.shape[0]))
                for mask in masks
            ]
        ).to(self.device)

        self.idx += 1

class HyperpartisanDataset(Dataset):
    def __init__(self, json_file, tokenizer: BertTokenizer, max_len):
        data = pd.read_json(json_file, orient='records')
        self.docs = []
        self.masks = []
        for text in data['text']:
            res = tokenizer.batch_encode_plus(text.split('[SEP]'),
                                              max_length=max_len,
                                              pad_to_max_length=True,
                                              add_special_tokens=True,
                                              return_tensors='pt')
            self.docs.append(res['input_ids'])
            self.masks.append(res['attention_mask'])

        self.y = torch.LongTensor((data['label'] == 'true').astype('int').to_numpy())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.docs[idx], self.masks[idx], self.y[idx]

def collate_pad_fn(batch):
    x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    return x_pad, LongTensor(y)
