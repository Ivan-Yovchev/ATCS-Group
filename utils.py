import os
import re
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def persuasiveness_scheduler(epoch, epochs):

    t = epoch/epochs

    # equally split time into sampling distances of 5, 4, 3, 2, and 1
    if t<=0.2:
        return [0, 5]
    elif t<=0.4:
        return random.sample(
            [
                [0,4],
                [1,5]
            ],
            1
        )[0]
    elif t<=0.6:
        return random.sample(
            [
                [0,3],
                [1,4],
                [2,5]
            ],
            1
        )[0]
    elif t<=0.8:
        return random.sample(
            [
                [0,2],
                [1,3],
                [2,4],
                [3,5]
            ],
            1
        )[0]
    else:
        return random.sample(
            [
                [0,1],
                [1,2],
                [2,3],
                [3,4],
                [4,5]
            ],
            1
        )[0]


def get_acc(preds, targets, binary=False):

    if binary:  # binary
        preds = (preds > 0.5).to(torch.long)
    else:  # multiclass
        preds = preds.argmax(dim=-1)
    return torch.mean((preds == targets).float()).item()


class AccumulatorF1:
    def __init__(self):
        self.true_positives = 0
        self.tot_positives = 0
        self.tot_true = 0
        self.epsilon = 1e-16

    def add(self, out, label):

        # for the binary tasks with 2 outputs coming from metalearning.py
        if len(out.shape) > 1 and out.shape[1] == 2:
            out = nn.functional.softmax(out, dim=1)[:, 1]

        pred = (out > 0.5).to(torch.long)
        if len(pred.shape) > 1:
            pred = pred.squeeze(1)
        self.true_positives += (pred * label).sum().item()
        self.tot_positives += pred.sum().item()
        self.tot_true += label.sum().item()

    def reduce(self):
        precision = self.true_positives / (self.tot_positives + self.epsilon)
        recall = self.true_positives / (self.tot_true + self.epsilon)
        f1 = (2 * precision * recall) / (precision + recall + self.epsilon)
        return precision, recall, f1


def load_model(path, conv_model, task_classifier, sent_embedder, finetune):
    state_dicts = torch.load(path)
    task_classifier.load_state_dict(state_dicts['task_classifier'])
    conv_model.load_state_dict(state_dicts['cnn_model'])
    model = construct_common_model(finetune, conv_model, sent_embedder)

    return model, task_classifier


def save_model(dataset_type, conv_model, bert_model, task_classifier, epoch, time_log, fold=None, model_dir='models'):
    filename = f"{dataset_type}.{time_log}.{fold}.pt" \
        if fold is not None else f"{dataset_type}.{time_log}.pt"
    with open(os.path.join(model_dir, filename), 'wb') as f:
        torch.save({
            'cnn_model': conv_model.state_dict(),
            'task_classifier': task_classifier.state_dict(),
            'bert_model': bert_model.state_dict(),
            'epoch': epoch
        }, f)


def get_summary_writer(args, time_log):
    """ Creates an instance of the tensorboard summary writer for the specified run """
    if args.dataset_type == 'gcdc':
        dataset_name = re.findall(r'/(\w+)_\w+.csv', args.train_path)[0]
        return SummaryWriter(f'runs/{dataset_name}_{time_log}')
    else:
        return SummaryWriter(f'runs/{args.dataset_type}_{time_log}')