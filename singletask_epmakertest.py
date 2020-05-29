import argparse
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel

from cnn_model import CNNModel
from common import Common
from doc_emb_models import BertManager
from metatrain import get_dataset_paths
from datasets import EpisodeMaker, BertPreprocessor
from multitask_train import TaskClassifier
from train import loss_task_factory, train_model, eval_model
import torch.nn as nn
from copy import deepcopy
import logging



#
# JUST A TEST SCRIPT. TODO will be removed.
#

class TaskClassifierGCDC(nn.Module):
    def __init__(self, in_features):
        super(TaskClassifierGCDC, self).__init__()
        # 3 classes for gcdc, 6 classes for persuas,
        # 1 for hyperpartisan and 1 for fake_news (sigmoid is used)
        self.linear = nn.Linear(in_features, 3)

    def forward(self, input):
        return self.linear(input)


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


def main(args):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    dataset_descriptors = get_dataset_paths(args.dataset_json)
    dataset_types = [k['name'] for k in dataset_descriptors]

    # Init Bert layer and Conv
    model, conv_model, sent_embedder = init_common(args, bert_model)

    task_classifier = TaskClassifierGCDC(conv_model.get_n_blocks() * args.n_filters)
    ep_maker = EpisodeMaker(
        bert_tokenizer,
        args.max_len,
        args.max_sent,
        model.cnn.get_max_kernel(),
        args.device,
        datasets=dataset_descriptors,
        sent_embedder=None if args.finetune else sent_embedder
    )

    print(ep_maker.datasets['gcdc'])

    task_classifier = task_classifier.to(args.device)
    model = model.to(args.device)
    optim = torch.optim.Adam(list(model.parameters()) + list(task_classifier.parameters()), lr=args.lr)

    import random
    logging.info('Multitask training starting.')
    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    # writer = SummaryWriter(f'runs/multitaskep_{time_log}')
    for batch_nr in range(args.n_epochs):
        optim.zero_grad()
        dataset_type = 'gcdc'
        one_batch_dataset = ep_maker.get_episode(dataset_type=dataset_type, n_train=args.train_size_support)[
            'support_set']

        binary, loss = loss_task_factory(dataset_type)

        train_acc, train_loss = train_model(model, task_classifier, one_batch_dataset, loss, optim, binary,
                                            disp_tqdm=False)
        # writer.add_scalar(f'Train/{dataset_type}/multi/accuracy', train_acc, batch_nr)
        # writer.add_scalar(f'Train/{dataset_type}/multi/loss', train_loss, batch_nr)

        logging.info("dataset_type %s, acc %.4f, loss %.4f", dataset_type, train_acc, train_loss)
        logging.debug("max of gradients of task_classifier: %f",
                      max(p.grad.max() for p in
                          task_classifier.parameters()))  # we take the max because the mean wouldn't be informative
        logging.debug("avg of gradients of model: %f",
                      max(p.grad.max() for p in model.parameters() if p.grad is not None))

    for i in range(4):
        binary, loss = loss_task_factory('gcdc')
        test_set = ep_maker.datasets['gcdc'][i]['test']
        test_set.batch_size = 1
        test_set.shuffle()
        test_set = BertPreprocessor(test_set, sent_embedder, conv_model.get_max_kernel(), device=args.device, batch_size=8)
        acc, loss, _ = eval_model(model, task_classifier, test_set, loss, binary, disp_tqdm=False)
        logging.info("%s: accuracy %.4f", test_set.file, acc)



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_len", type=int, default=15, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=15, help="Max number of sentences in a doc")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training model")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--dataset_json", type=str, default='./dataset-paths.json',
                        help="JSON file containing the dataset paths")

    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--n_epochs_singletask", type=int, default=5, help="Number of shots for single-task adaptation")
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False,
                        help="Set to true to fine tune bert")
    parser.add_argument("--train_size_support", type=int, default=8, help="Size of support set during training")
    parser.add_argument("--train_size_query", type=int, default=8, help="Size of query set during training")
    parser.add_argument("--shots", type=int, default=8, help="Number of examples during meta validation/testing")
    parser.add_argument("--kernels", type=lambda x: [int(i) for i in x.split(',')], default="2,4,6",
                        help="Kernel sizes per cnn block")

    args = parser.parse_args()
    args.embed_size = 768
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    main(args)
