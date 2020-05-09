import os
import pandas as pd

from torch import LongTensor, stack, float32 as tfloat32, squeeze
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from transformers import BertTokenizer

from random import shuffle
from math import ceil


def get_dataset(dataset_type, path, tokenizer, max_len, batch_size, device):
    if dataset_type == "gcdc":
        return GCDC_Dataset(path, tokenizer, max_len, batch_size, 'text', '\n\n', device)
    elif dataset_type == "hyperpartisan":
        return HyperpartisanDataset(path, tokenizer, max_len, batch_size, 'text', '[SEP]', device)
    elif dataset_type == "persuasiveness":
        return PersuasivenessDataset(path, tokenizer, max_len, batch_size, 'Justification', '[SEP]', device)
    elif dataset_type == "fake_news":
        return FakeNewsDataset(path, tokenizer, max_len, batch_size, 'text', '[SEP]', device)
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')


class ParentDataset(Dataset):
    """docstring for ParentDataset"""

    def __init__(self, file, tokenizer: BertTokenizer, max_len, batch_size, field_id, split_token, device):
        super(ParentDataset, self).__init__()
        self.file = file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.field_id = field_id
        self.split_token = split_token
        self.device = device

    def shuffle(self):
        # Shuffle docs
        temp = list(zip(self.docs, self.masks, self.y))
        shuffle(temp)

        self.docs, self.masks, self.y = zip(*temp)

        self.idx = 0

    def get_data(self, data):
        docs = []
        masks = []
        for text in data[self.field_id]:
            res = self.tokenizer.batch_encode_plus(text.split(self.split_token),
                                                   max_length=self.max_len,
                                                   pad_to_max_length=True,
                                                   add_special_tokens=True,
                                                   return_tensors='pt')
            docs.append(res['input_ids'])
            masks.append(res['attention_mask'])

        return docs, masks

    def __len__(self):
        return ceil(len(self.y) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):

        if self.idx == 0:
            self.shuffle()
        elif self.idx >= len(self):
            self.idx = 0
            raise StopIteration

        # Get interval of current batch
        idx_start, idx_end = self.batch_size * self.idx, min(self.batch_size * (self.idx + 1), len(self.y))

        # Get batch lists
        docs, masks, ys = self.docs[idx_start:idx_end], self.masks[idx_start:idx_end], self.y[idx_start:idx_end]

        # Pad docs (to have the same number of sentences)
        max_sent = max((doc.shape[0] for doc in docs))

        docs = stack(
            [
                F.pad(doc, pad=(0, 0, 0, max_sent - doc.shape[0]))
                for doc in docs
            ]
        ).to(self.device)

        masks = stack(
            [
                F.pad(mask, pad=(0, 0, 0, max_sent - mask.shape[0]))
                for mask in masks
            ]
        ).to(self.device)

        self.idx += 1

        return docs, masks, LongTensor(ys).to(self.device)


class GCDC_Dataset(ParentDataset):

    def __init__(self, file, tokenizer: BertTokenizer, max_len, batch_size, field_id, split_token, device):
        super(GCDC_Dataset, self).__init__(file, tokenizer, max_len, batch_size, field_id, split_token, device)

        data = pd.read_csv(self.file)
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor(data['labelA'] - 1)

        self.shuffle()


class FakeNewsDataset(ParentDataset):
    def __init__(self, file, tokenizer: BertTokenizer, max_len, batch_size, field_id, split_token, device):
        super(FakeNewsDataset, self).__init__(file, tokenizer, max_len, batch_size, field_id, split_token, device)

        data = pd.read_csv(self.file, sep='\t', header=0, names=['text', 'label'])

        self.docs, self.masks = self.get_data(data)

        self.y = LongTensor(data['label'].to_numpy())

        self.shuffle()


class HyperpartisanDataset(ParentDataset):

    def __init__(self, file, tokenizer: BertTokenizer, max_len, batch_size, field_id, split_token, device):
        super(HyperpartisanDataset, self).__init__(file, tokenizer, max_len, batch_size, field_id, split_token, device)

        data = pd.read_json(self.file, orient='records')
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor((data['label'] == 'true').astype('int').to_numpy())

        self.shuffle()


class PersuasivenessDataset(ParentDataset):

    def __init__(self, file, tokenizer: BertTokenizer, max_len, batch_size, field_id, split_token, device):
        super(PersuasivenessDataset, self).__init__(file, tokenizer, max_len, batch_size, field_id, split_token, device)

        data = pd.read_json(self.file, orient='records')
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor(data['Persuasiveness'].to_numpy())

        self.shuffle()


def collate_pad_fn(batch):
    x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    return x_pad, LongTensor(y)
