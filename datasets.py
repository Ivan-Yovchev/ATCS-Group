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

    def get_classes(self):
        return set([x.item() for x in self.y])

    def get_n_classes(self):
        return len(self.get_classes())

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


class PersuasivenessDataset(ParentDataset):

    def __init__(self, file, tokenizer: BertTokenizer, max_len, max_sent, batch_size, field_id, split_token, device):
        super(PersuasivenessDataset, self).__init__(file, tokenizer, max_len, max_sent, batch_size, field_id,
                                                    split_token, device)

        data = pd.read_json(self.file, orient='records')
        self.docs, self.masks = self.get_data(data)
        self.y = LongTensor(data['Persuasiveness'].to_numpy() - 1)

        self.shuffle()


class BertPreprocessor(ParentDataset):

    def __init__(self, decorated, encoder, max_kernel, batch_size=1, device = None):
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
        batch = np.array([np.hstack((i, np.zeros((dim, pad-i.shape[1])))) for i in samples])
        
        return [Tensor(batch).to(self.device)], LongTensor(self.y[idx_start:idx_end]).to(self.device)

class Scheduler(object):
    """scheduler to pick classes and keep track of epoch for EpisodeMaker"""

    def __init__(self, epochs, sampler):

        self.epoch = 0
        self.epochs = epochs
        self.sampler = sampler

    def __iter__(self):
        return self

    def __next__(self):

        if self.epoch >= self.epochs:
            raise StopIteration

        classes = self.sampler(self.epoch, self.epochs)

        self.epoch += 1
        return classes


class EpisodeMaker(object):
    """docstring for EpisodeMaker"""

    def __init__(self, tokenizer: BertTokenizer, max_len, max_sent, max_kernel, device, datasets=[],
                 gcdc_ext=["Clinton", "Enron", "Yahoo", "Yelp"], sent_embedder=None):
        super(EpisodeMaker, self).__init__()

        self.sent_embedder = sent_embedder
        self.device = device
        self.max_kernel = max_kernel
        self.cpu_device = torch.device("cpu")

        assert (len(datasets) != 0)

        self.datasets = {}
        for dataset in datasets:
            key = dataset["name"]

            if key != "gcdc":
                self.datasets[key] = [{
                    "train": get_dataset(key, dataset["train"], tokenizer, max_len, max_sent, 1, self.cpu_device),
                    "test": get_dataset(key, dataset["test"], tokenizer, max_len, max_sent, 1, self.cpu_device)
                }]
            else:
                path = dataset["train"]
                if ".csv" in path:
                    path = os.path.dirname(path)

                self.datasets[key] = []
                for ext in gcdc_ext:
                    sub_gcdc = {
                        "train": get_dataset(key, path + ext + "_train.csv", tokenizer, max_len, max_sent, 1, self.cpu_device),
                        "test": get_dataset(key, path + ext + "_test.csv", tokenizer, max_len, max_sent, 1, self.cpu_device)
                    }

                    self.datasets[key].append(sub_gcdc)

    def get_episode(self, dataset_type, n_train=8, n_test=8, classes_sampled="all", batch_size_when_all=32):

        dataset = random.sample(self.datasets[dataset_type], 1)[0]

        classes_tot = list(range(dataset["train"].get_n_classes()))

        # OPTIMIZE: maybe not true
        # assume classes are always 0 .. n
        if isinstance(classes_sampled, str) and classes_sampled == "all":
            allowed_classes = classes_tot
        elif isinstance(classes_sampled, Scheduler):
            allowed_classes = next(classes_sampled)
        elif isinstance(classes_smapled, (list, tuple)):
            allowed_classes = classes_sampled
        else:
            if not isinstance(classes_sampled, int):
                k = floor(len(classes_tot*classes_sampled))
            else:
                k = classes_sampled
            allowed_clases = random.sample(classes_tot, k)

        support_set = self.__sample_dataset(dataset["train"], allowed_classes, n_train)

        if isinstance(n_test, str) and n_test == "all":
            query_set = dataset["test"] if self.sent_embedder is None else BertPreprocessor(dataset["test"], self.sent_embedder, self.max_kernel, batch_size_when_all, self.device)
        else:
            query_set = self.__sample_dataset(dataset["test"], allowed_classes, n_test)

        return {
            "support_set": support_set,
            "query_set": query_set
        }

    def __sample_dataset(self, split, allowed_classes, k):

        pars = split.file, split.tokenizer, split.max_len, split.max_sent, split.batch_size, split.field_id, split.split_token, split.device

        split.shuffle()
        split = list(zip(split.docs, split.masks, split.y))
        split = list(filter(lambda x: x[2] in allowed_classes, split))

        # Relabel remaining classes into progressive indices

        relabel = {k : v for v,k in enumerate(allowed_classes)}

        split = list(
            map(
                lambda x : (x[0], x[1], x[2]*0 + relabel[x[2].item()]),
                split
            )
        )

        # Sample datapoints per class

        final_split = []
        sample_size = round(k / len(allowed_classes))

        for class_y in range(len(allowed_classes)):

            if class_y == len(allowed_classes) - 1:
                sample_size = k - class_y * sample_size

            population = list(filter(lambda x: x[2] == class_y, split))
            final_split += random.sample(population, min(sample_size, len(population)))

        shuffle(final_split)

        dataset = Manual_Dataset(
            *zip(*final_split),
            *pars
        )

        return dataset if self.sent_embedder is None else BertPreprocessor(dataset, self.sent_embedder, self.max_kernel, k, self.device)


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
    model = EpisodeMaker(bert_tokenizer, 50, 200, 'cpu', [a, b])
    temp = model.get_episode('gcdc')
