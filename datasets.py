import os
import pandas as pd
import random
import numpy as np

from torch import LongTensor, stack, float32 as tfloat32, squeeze
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from transformers import BertTokenizer

from random import shuffle
from math import ceil


def get_dataset(dataset_type, path, tokenizer, max_len, max_sent, batch_size, device, hyperpartisan_10fold = False):
    if dataset_type == "gcdc":
        return GCDC_Dataset(path, tokenizer, max_len, max_sent, batch_size, 'text', '\n\n', device)
    elif dataset_type == "hyperpartisan":
        if hyperpartisan_10fold:
            return Hyperpartisan10Fold(path, tokenizer, max_len, max_sent, batch_size, 'text', '[SEP]', device)
        else:
            return HyperpartisanDataset(path, tokenizer, max_len, max_sent, batch_size, 'text', '[SEP]', device)
    elif dataset_type == "persuasiveness":
        return PersuasivenessDataset(path, tokenizer, max_len, max_sent, batch_size, 'Justification', '[SEP]', device)
    elif dataset_type == "fake_news":
        return FakeNewsDataset(path, tokenizer, max_len, max_sent, batch_size, 'text', '[SEP]', device)
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')


class ParentDataset(Dataset):
    """docstring for ParentDataset"""

    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device):
        super(ParentDataset, self).__init__()
        self.file = file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_sent = max_sent
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

    def get_n_classes(self):

        return len(set([x.item() for x in self.y]))

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
        max_sent = min(max_sent, self.max_sent)

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

    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device):
        super(GCDC_Dataset, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id, split_token,
                                           device)

        data = pd.read_csv(self.file)
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor(data['labelA'] - 1)

        self.shuffle()


class FakeNewsDataset(ParentDataset):
    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device):
        super(FakeNewsDataset, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id, split_token,
                                              device)

        data = pd.read_csv(self.file, sep='\t', header=0, names=['text', 'label'])

        self.docs, self.masks = self.get_data(data)

        self.y = LongTensor(data['label'].to_numpy())

        self.shuffle()


class HyperpartisanDataset(ParentDataset):

    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device):
        super(HyperpartisanDataset, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id,
                                                   split_token, device)

        data = pd.read_json(self.file, orient='records')
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor((data['label'] == 'true').astype('int').to_numpy())

        self.shuffle()


class Hyperpartisan10Fold:
    def __init__(self, *varg):
        file = varg[0]
        self.varg = varg
        self.data = pd.read_json(file, orient='records')
        self.k = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.k >= 10:
            raise StopIteration
        length = len(self.data)
        step = length // 10
        start = self.k * step
        end = start + step
        mask = (start <= np.arange(length)) * (np.arange(length) < end)
        train_set = self.data.iloc[~mask]
        test_set = self.data.iloc[mask]
        self.k += 1
        return HyperpartisanDatasetFold(train_set, *self.varg), HyperpartisanDatasetFold(test_set, *self.varg)


class HyperpartisanDatasetFold(ParentDataset):
    def __init__(self, data, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token,
                 device):
        super(HyperpartisanDatasetFold, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id,
                                                       split_token, device)

        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor((data['label'] == 'true').astype('int').to_numpy())
        self.shuffle()


class PersuasivenessDataset(ParentDataset):

    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device):
        super(PersuasivenessDataset, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id,
                                                    split_token, device)

        data = pd.read_json(self.file, orient='records')
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor(data['Persuasiveness'].to_numpy() - 1)

        self.shuffle()


class EpisodeMaker(object):
    """docstring for EpisodeMaker"""

    def __init__(self, tokenizer: BertTokenizer, max_len, device, datasets=[],
                 gcdc_ext=["Clinton", "Enron", "Yahoo", "Yelp"]):
        super(EpisodeMaker, self).__init__()

        assert (len(datasets) != 0)

        self.datasets = {}
        for dataset in datasets:
            key = dataset["name"]

            if key != "gcdc":
                self.datasets[key] = [{
                    "train": get_dataset(key, dataset["train"], tokenizer, max_len, 1, device),
                    "test": get_dataset(key, dataset["test"], tokenizer, max_len, 1, device)
                }]
            else:
                path = dataset["train"]
                if ".csv" in path:
                    path = os.path.dirname(path)

                self.datasets[key] = []
                for ext in gcdc_ext:
                    sub_gcdc = {
                        "train": get_dataset(key, path + ext + "_train.csv", tokenizer, max_len, 1, device),
                        "test": get_dataset(key, path + ext + "_test.csv", tokenizer, max_len, 1, device)
                    }

                    self.datasets[key].append(sub_gcdc)

    def get_episode(self, dataset_type, classes=2, n_train=8, n_test=4):

        if n_train % classes != 0:
            n_train -= n_train % classes

        if n_test % classes != 0:
            n_test -= n_test % classes

        dataset = random.sample(self.datasets[dataset_type], 1)[0]

        n_classes = dataset["train"].get_n_classes()

        allowed_classes = []
        for _ in range(classes):
            class_idx = random.randint(0, n_classes - 1)

            while class_idx in allowed_classes:
                class_idx = random.randint(0, n_classes - 1)

            allowed_classes.append(class_idx)

        return {
            "support_set": self.__sample_dataset(dataset["train"], allowed_classes, n_train),
            "query_set": self.__sample_dataset(dataset["test"], allowed_classes, n_test)
        }

    def __sample_dataset(self, split, allowed_classes, k):

        split.shuffle()
        split = list(zip(split.docs, split.masks, split.y))
        split = list(filter(lambda x: x[2] in allowed_classes, split))

        final_split = []
        for i in range(len(allowed_classes)):
            final_split += random.sample(list(filter(lambda x: x[2] == allowed_classes[i], split)),
                                         int(k / len(allowed_classes)))

        shuffle(final_split)
        docs, masks, y = zip(*final_split)

        return (docs, masks, y)


def collate_pad_fn(batch):
    x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    return x_pad, LongTensor(y)


if __name__ == "__main__":
    from transformers import BertModel, BertTokenizer

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    a = {"name": "gcdc", "train": "./data/GCDC/", "test": "./data/GCDC"}
    b = {"name": "persuasiveness", "train": "./data/DebatePersuasiveness/persuasiveness_dataset-train.json",
         "test": "./data/DebatePersuasiveness/persuasiveness_dataset-test.json"}
    model = EpisodeMaker(bert_tokenizer, 200, 'cpu', [a, b])
    model.get_episode('gcdc', 1)
