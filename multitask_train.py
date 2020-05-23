import argparse
from typing import Tuple

import torch
from transformers import BertTokenizer, BertModel

from cnn_model import CNNModel
from common import Common
from doc_emb_models import BertManager
from metatrain import get_dataset_paths
from datasets import EpisodeMaker
from train import loss_task_factory, train_model
import torch.nn as nn
from copy import deepcopy
import logging


class TaskClassifier(nn.Module):
    def __init__(self, in_features):
        super(TaskClassifier, self).__init__()
        # 3 classes for gcdc, 6 classes for persuas,
        # 1 for hyperpartisan and 1 for fake_news (sigmoid is used)
        self.linear = nn.Linear(in_features, 11)

    def forward(self, input):
        out = self.linear(input)
        # the last two outputs are for two binary tasks, they need to be sigmoided
        out[:, 9:] = torch.sigmoid(out[:, 9:])
        return out


# TODO will change if we change the classes on which we run it
class TaskClassifierWrapper(nn.Module):
    """It allows to pass the task-classifier with multiclass output to the standard training procedure"""

    def __init__(self, task_classifier: TaskClassifier, dataset_type: str):
        super(TaskClassifierWrapper, self).__init__()
        self.task_classifier = task_classifier
        self.start, self.length = self._retrieve_interval(dataset_type)

    def forward(self, input):
        out = self.task_classifier(input)
        return torch.narrow(out, 1, self.start, self.length)

    @staticmethod
    def _retrieve_interval(dataset_type: str) -> Tuple[int, int]:
        # TODO maybe let's not hardcode it?
        # contains the relevant portions of each task in the task_classifier output
        if dataset_type == 'gcdc':
            return 0, 3
        if dataset_type == 'persuasiveness':
            return 3, 6
        if dataset_type == 'hyperpartisan':
            return 9, 1
        if dataset_type == 'fake_news':
            return 10, 1
        raise ValueError('{} is not recognized as a dataset type'.format(dataset_type))


def main(args):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    dataset_descriptors = get_dataset_paths(args.dataset_json)
    dataset_types = [k['name'] for k in dataset_descriptors]

    # Init Bert layer and Conv
    model, conv_model, sent_embedder = init_common(args, bert_model)

    task_classifier = TaskClassifier(conv_model.get_n_blocks() * args.n_filters)
    ep_maker = EpisodeMaker(
        bert_tokenizer,
        args.max_len,
        args.max_sent,
        model.cnn.get_max_kernel(),
        args.device,
        datasets=dataset_descriptors,
        sent_embedder=None if args.finetune else sent_embedder
    )

    task_classifier = task_classifier.to(args.device)
    model = model.to(args.device)
    optim = torch.optim.SGD(list(model.parameters()) + list(task_classifier.parameters()), lr=args.lr)

    logging.info('Multitask training starting.')
    for batch_nr in range(args.n_epochs):
        optim.zero_grad()
        dataset_type = dataset_types[batch_nr % 4]
        one_batch_dataset = ep_maker.get_episode(dataset_type=dataset_type)['support_set']
        binary, loss = loss_task_factory(dataset_type)
        tcw = TaskClassifierWrapper(task_classifier, dataset_type)

        res = train_model(model, tcw, one_batch_dataset, loss, optim, binary, disp_tqdm=False)
        logging.info("dataset_type %s, acc %.4f, loss %.4f", dataset_type, *res)
        logging.debug("max of gradients of task_classifier: %f",
                      max(p.grad.max() for p in task_classifier.parameters()))
        logging.debug("avg of gradients of model: %f",
                      sum(p.grad.mean() for p in model.parameters() if p.grad is not None))

        # this is the general copy of the model
    trained_general_model = (model.cpu(), task_classifier.cpu())
    # for dataset_type in dataset_types:
    #     logging.info('training task-specific mode on %s', dataset_type)
    #     # model_sp, conv_model, _ = init_common(args, bert_model)
    #     # task_classifier_sp = TaskClassifier(conv_model.get_n_blocks() * args.n_filters)
    #     model_sp, task_classifier_sp = deepcopy(trained_general_model)
    #     model_sp = model_sp.to(args.device)
    #     task_classifier_sp = task_classifier_sp.to(args.device)
    #
    #     batch = ep_maker.get_episode(dataset_type=dataset_type)['support_set']
    #     X, y = next(iter(batch))
    #


def init_common(args, bert_model):
    sent_embedder = BertManager(bert_model, args.device)
    conv_model = CNNModel(args.embed_size, torch.device("cpu"), n_filters=args.n_filters, filter_sizes=args.kernels,
                          batch_norm_eval=True)
    # Build unified model
    model = Common(
        conv_model,
        conv_model.get_n_blocks() * args.n_filters,
        encoder=sent_embedder if args.finetune else lambda x: x,
    )
    return model, conv_model, sent_embedder


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_len", type=int, default=15, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=15, help="Max number of sentences in a doc")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for training model")
    # parser.add_argument("--meta_lr", type=float, default=0.0001, help="Learning rate for updateing meta model")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--dataset_json", type=str, default='./dataset-paths.json',
                        help="JSON file containing the dataset paths")

    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--n_epochs_singletask", type=int, default=5, help="Number of shots for single-task adaptation")
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False,
                        help="Set to true to fine tune bert")
    parser.add_argument("--train_size_support", type=int, default=8, help="Size of support set during training")
    parser.add_argument("--train_size_query", type=int, default=8, help="Size of query set during training")
    # parser.add_argument("--shots", type=int, default=8, help="Number of examples during meta validation/testing")
    parser.add_argument("--kernels", type=lambda x: [int(i) for i in x.split(',')], default="2,4,6",
                        help="Kernel sizes per cnn block")

    args = parser.parse_args()
    args.embed_size = 768
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    main(args)
