import os
import pandas as pd
import random
import numpy as np
import torch

from torch import LongTensor, stack, float32 as tfloat32, squeeze, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from transformers import BertTokenizer

from random import shuffle
from math import ceil
import json


def get_dataset(dataset_type, path, tokenizer, max_len, max_sent, batch_size, device, hyperpartisan_10fold=False):
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
            sents = sorted(text.split(self.split_token), key=lambda s: len(s), reverse=True)
            res = self.tokenizer.batch_encode_plus(sents,
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

        docs = [doc[:max_sent] for doc in docs]
        masks = [mask[:max_sent] for mask in masks]

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

        return (docs, masks), LongTensor(ys).to(self.device)


class Manual_Dataset(ParentDataset):

    def __init__(self, docs, masks, ys, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id,
                 split_token, device):
        super().__init__(file, tokenizer, max_len, max_sent, batch_size, field_id, split_token, device)

        self.docs = docs
        self.masks = masks
        self.y = ys

        self.shuffle()


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

    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device,
                 data=None):
        super(HyperpartisanDataset, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id,
                                                   split_token, device)

        if data is None:
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
        return HyperpartisanDataset(*self.varg, data=train_set), HyperpartisanDataset(*self.varg, data=test_set)

    def to(self, device):
        self.device = device


class PersuasivenessDataset(ParentDataset):

    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device):
        super(PersuasivenessDataset, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id,
                                                    split_token, device)
        data = pd.read_json(self.file, orient='records')
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor(data['Persuasiveness'].to_numpy() - 1)
        self.shuffle()


class NumpyBackedDataset(Dataset):
    def __init__(self, filename_prefix, device, create=False, numpy_features=None, numpy_labels=None):
        self.max_shard_size = 1024
        if create:
            length = len(numpy_features)
            n_shards = int(np.ceil(length / self.max_shard_size))
            # indices = np.arange(1, n_shards) * self.max_shard_size

            for shard in range(n_shards):
                start = shard * self.max_shard_size
                end = min(length, start + self.max_shard_size)

                X = numpy_features[start:end]
                max_length = max(arr.shape[2] for arr in X)
                X_new = []
                lengths = []
                for x in X:
                    l = x.shape[2]
                    X_new.append(np.pad(x, pad_width=((0, 0), (0, 0), (0, max_length - l))))
                    lengths.append(l)

                Y = numpy_labels[start:end]
                np.savez(filename_prefix + f".{shard:02d}.npz", X=np.array(X_new), L=np.array(lengths), Y=Y)

            # indices = indices.tolist()
            with open(filename_prefix + '.json', 'w') as f:
                json.dump({"len": length, "n_shards": n_shards}, f)
            del numpy_features, numpy_labels

        with open(filename_prefix + '.json', 'r') as f:
            k = json.load(f)
            self.length = k['len']
            # self.indices = np.array(k['indices'])
            self.n_shards = k['n_shards']

        self.shards = []
        for shard in range(self.n_shards):
            self.shards.append(np.load(filename_prefix + f".{shard:02d}.npz", mmap_mode='r'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        shard = idx // self.max_shard_size
        shardwise_idx = idx % self.max_shard_size
        true_length = self.shards[shard]['L'][shardwise_idx]
        x = torch.from_numpy(self.shards[shard]['X'][shardwise_idx][:, :, :true_length]).squeeze(dim=0)
        y = torch.tensor(self.shards[shard]['Y'][shardwise_idx]).squeeze(dim=0)
        return x, y

    @staticmethod
    def collate_fn(batch):
        X, Y = zip(*batch)
        # need to make the max-length at least the minimum kernelgit
        max_length = max(6, max(arr.shape[1] for arr in X))
        X_new = [np.pad(x, pad_width=((0, 0), (0, max_length - x.shape[1]))) for x in X]
        X_new = np.array(X_new)
        Y = np.array(Y)
        return torch.from_numpy(X_new), torch.from_numpy(Y)


class BertPreprocessor(ParentDataset):

    def __init__(self, decorated, encoder, max_kernel, batch_size=1, device=None):
        super(BertPreprocessor, self).__init__(
            decorated.file,
            decorated.tokenizer,
            decorated.max_len,
            decorated.max_sent,
            decorated.batch_size,
            decorated.field_id,
            decorated.split_token,
            decorated.device
        )

        self.device = decorated.device if device is None else device
        self.batch_size = batch_size
        self.max_kernel = max_kernel

        self.docs = []

        for ((doc, mask), _) in decorated:
            self.docs.append(np.squeeze(encoder(doc, mask).cpu().detach().numpy(), axis=0))

        self.y = np.array(decorated.y)
        self.shuffle()

    def shuffle(self):
        # Shuffle embeddings
        temp = list(range(len(self.y)))
        shuffle(temp)

        self.docs = [self.docs[i] for i in temp]
        self.y = self.y[temp]

        self.idx = 0

    def get_data(self, data):
        pass

    def __next__(self):

        if self.idx == 0:
            self.shuffle()
        elif self.idx >= len(self):
            self.idx = 0
            raise StopIteration

        # Get interval of current batch
        idx_start, idx_end = self.batch_size * self.idx, min(self.batch_size * (self.idx + 1), len(self.y))
        self.idx += 1

        # sample jagged batch
        samples = self.docs[idx_start:idx_end]

        # get embeddings dim
        dim = self.docs[0].shape[0]

        # get max length in batch
        pad = max([doc.shape[1] for doc in samples])

        # make sure max length is not smaller than largest kernel
        if pad < self.max_kernel:
            pad = self.max_kernel

        # pad batch
        batch = np.array([np.hstack((i, np.zeros((dim, pad - i.shape[1])))) for i in samples])

        return Tensor(batch).to(self.device), LongTensor(self.y[idx_start:idx_end]).to(self.device)


class EpisodeMaker(object):
    """docstring for EpisodeMaker"""

    def __init__(self, tokenizer: BertTokenizer, max_len, max_sent, device, datasets=[],
                 gcdc_ext=["Clinton", "Enron", "Yahoo", "Yelp"], sent_embedder=None):
        super(EpisodeMaker, self).__init__()

        self.sent_embedder = sent_embedder

        assert (len(datasets) != 0)

        self.datasets = {}
        for dataset in datasets:
            key = dataset["name"]

            if key != "gcdc":
                self.datasets[key] = [{
                    "train": get_dataset(key, dataset["train"], tokenizer, max_len, max_sent, 1, device),
                    "test": get_dataset(key, dataset["test"], tokenizer, max_len, max_sent, 1, device)
                }]
            else:
                path = dataset["train"]
                if ".csv" in path:
                    path = os.path.dirname(path)

                self.datasets[key] = []
                for ext in gcdc_ext:
                    sub_gcdc = {
                        "train": get_dataset(key, path + ext + "_train.csv", tokenizer, max_len, max_sent, 1, device),
                        "test": get_dataset(key, path + ext + "_test.csv", tokenizer, max_len, max_sent, 1, device)
                    }

                    self.datasets[key].append(sub_gcdc)

    def get_episode(self, dataset_type, n_train=8, n_test=4):

        dataset = random.sample(self.datasets[dataset_type], 1)[0]

        n_classes = dataset["train"].get_n_classes()

        # OPTIMIZE: maybe not true
        # assume classes are always 0 .. n
        allowed_classes = [idx for idx in range(n_classes)]

        return {
            "support_set": self.__sample_dataset(dataset["train"], allowed_classes, n_train),
            "query_set": self.__sample_dataset(dataset["test"], allowed_classes, n_test)
        }

    def __sample_dataset(self, split, allowed_classes, k):

        pars = split.file, split.tokenizer, split.max_len, split.max_sent, split.batch_size, split.field_id, split.split_token, split.device

        split.shuffle()
        split = list(zip(split.docs, split.masks, split.y))
        split = list(filter(lambda x: x[2] in allowed_classes, split))

        final_split = []
        sample_size = round(k / len(allowed_classes))

        for i, class_y in enumerate(allowed_classes):

            if i == len(allowed_classes) - 1:
                sample_size = k - i * sample_size

            final_split += random.sample(list(filter(lambda x: x[2] == class_y, split)), sample_size)

        shuffle(final_split)

        dataset = Manual_Dataset(
            *zip(*final_split),
            *pars
        )

        return dataset if self.sent_embedder is None else BertPreprocessor(dataset, self.sent_embedder, batch_size=k)


def just_apply_bert(dataset: ParentDataset, bert):
    with torch.no_grad():
        X = []
        for (d, m), _ in dataset:
            X.append(bert(d, m).cpu())
        # X = torch.stack(X).numpy()
        Y = np.array(dataset.y)
    return X, Y


if __name__ == "__main__":
    from transformers import BertModel, BertTokenizer

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    a = {"name": "gcdc", "train": "./data/GCDC/", "test": "./data/GCDC"}
    b = {"name": "persuasiveness", "train": "./data/DebatePersuasiveness/persuasiveness_dataset-train.json",
         "test": "./data/DebatePersuasiveness/persuasiveness_dataset-test.json"}
    model = EpisodeMaker(bert_tokenizer, 50, 200, 'cpu', [a, b])
    temp = model.get_episode('gcdc')
