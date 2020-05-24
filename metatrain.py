import torch
import transformers

from tqdm import tqdm
from torch import nn
from transformers import BertModel, BertTokenizer
from datasets import EpisodeMaker, Scheduler
from utils import persuasiveness_scheduler
from doc_emb_models import *
from cnn_model import CNNModel
from train import train_model, eval_model
from common import Common
import random
from copy import deepcopy
import argparse
import json


class Task:
    '''
    contains all data needed with regards to a specific task
    '''

    def __init__(self, get_episode, loss, n_classes):
        assert callable(get_episode)
        self.get_episode = get_episode  # should be a lambda
        self.n_classes = n_classes
        self.loss = loss


def train_support(model: nn.Module, task: Task, init_optim, n_train=8, n_test=8):
    # Get episode
    ep = task.get_episode(n_train, n_test)

    # Construct output layer
    task_classifier, protos = model.get_outputlayer(ep["support_set"])

    # Construct copy of doc_embedder to simulate training on current task
    model_cp = deepcopy(model)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_cp.to(device)
    task_classifier.to(device)

    # Initialize optimizer for copy
    optim = init_optim(list(model_cp.parameters()) + list(task_classifier.parameters()))

    # import pdb
    # pdb.set_trace()
    train_model(model_cp, task_classifier, ep["support_set"], task.loss, optim, False, False)

    # Step 6 from FO-Proto MAML pdf
    protos = protos.to(task_classifier.weight.device)
    task_classifier.weight = nn.Parameter(protos + (task_classifier.weight - protos).detach())

    # Get gradients for main (original) model update

    # Reset gradients
    optim.zero_grad()

    return model_cp, ep, task_classifier


def run_task_batch(model: nn.Module, tasks, init_optim, lr, n_train=8, n_test=8, n_episodes = 1):
    class EmptyOptim:

        def step(self):
            return

        def zero_grad(self):
            return

    empty_optim = EmptyOptim()

    # Store gradients for each simulated model (trained on each task in the given batch)
    meta_grads = {}

    for task in tasks:

        for _ in range(n_episodes):

            model_cp, ep, task_classifier = train_support(model, task, init_optim, n_train, n_test)

            # Train on task episode
            train_model(model_cp, task_classifier, ep["query_set"], task.loss, empty_optim, False, False)

            # Accumulate gradients (per task)
            for par_name, par in dict(list(model_cp.named_parameters())).items():

                if par.grad is None:
                    continue

                grad = par.grad / len(ep["query_set"])

                if par_name not in meta_grads:
                    meta_grads[par_name] = grad
                else:
                    meta_grads[par_name] += grad

            del model_cp
            del task_classifier
            del ep
            torch.cuda.empty_cache()

    # Apply gradients
    with torch.no_grad():
        for par_name, par in dict(list(model.named_parameters())).items():
            par -= lr * meta_grads[par_name].cpu()

    model.zero_grad()


def meta_valid(model: nn.Module, task: Task, init_optim, support_set_size=8, query_set_size=8):
    model_cp, ep, task_classifier = train_support(model, task, init_optim, n_train=support_set_size,
                                                  n_test=query_set_size)
    results = eval_model(model_cp, task_classifier, ep["query_set"], task.loss, False, False)

    del model_cp
    del ep
    del task_classifier
    torch.cuda.empty_cache()

    return results


def main(args):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Define tasks episodes
    fake_news_desc, gcdc_desc, partisan_desc, pers_desc = get_dataset_paths(args.dataset_json)

    # Init Bert layer
    sent_embedder = BertManager(bert_model, args.device)

    # Init Conv layer
    conv_model = CNNModel(args.embed_size, torch.device("cpu"), n_filters=args.n_filters, filter_sizes=args.kernels)

    # Build unified model
    model = Common(
        conv_model,
        conv_model.get_n_blocks() * args.n_filters,
        encoder=sent_embedder if args.finetune else lambda x: x,
    )

    ep_maker = EpisodeMaker(
        bert_tokenizer,
        args.max_len,
        args.max_sent,
        model.cnn.get_max_kernel(),
        args.device,
        datasets=[gcdc_desc, pers_desc, partisan_desc, fake_news_desc],
        sent_embedder=None if args.finetune else sent_embedder
    )

    # Define tasks
    gcdc = Task(
        lambda m=8, n=8: ep_maker.get_episode('gcdc', n_train=m, n_test=n),
        nn.CrossEntropyLoss(),
        3
    )

    scheduler = Scheduler(
        epochs=args.meta_epochs*3,
        sampler=persuasiveness_scheduler
    )

    persuasiveness = Task(
        lambda m=8, n=8: ep_maker.get_episode(
            'persuasiveness',
            n_train=m,
            n_test=n,
            classes_sampled=scheduler,
        ),
        nn.CrossEntropyLoss(),
        6
    )

    partisan = Task(
        lambda m=8, n=8: ep_maker.get_episode('hyperpartisan', n_train=m, n_test=n),
        nn.CrossEntropyLoss(),
        2
    )

    fake_news = Task(
        lambda m=8, n=8: ep_maker.get_episode('fake_news', n_train=m, n_test=n),
        nn.CrossEntropyLoss(),
        2
    )

    # Define optimizer constructor
    init_optim = lambda pars: transformers.optimization.AdamW(pars, args.lr)

    best_acc = None
    best_model = None

    # meta train
    display_log = tqdm(range(args.meta_epochs), total=0, position=1, bar_format='{desc}')
    for i in tqdm(range(args.meta_epochs), desc="Meta-epochs", total=args.meta_epochs, position=0):
        run_task_batch(model, random.choices([gcdc, persuasiveness], k=args.meta_batch), init_optim, args.meta_lr, n_train=args.train_size_support,
                       n_test=args.train_size_query, n_episodes = args.n_episodes)

        # Meta Validation
        acc, loss = meta_valid(model, partisan, init_optim, support_set_size=args.shots, query_set_size='all')

        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model)

        display_log.set_description_str(f"Meta-valid {i:02d} acc: {acc:.4f} loss: {loss:.4f}")
    display_log.close()

    # meta test
    acc, loss = meta_valid(best_model, fake_news, init_optim, support_set_size=args.shots, query_set_size='all')
    print("Final: ", acc, loss)


def get_dataset_paths(dataset_json_file):
    with open(dataset_json_file, 'r') as f:
        dataset_desc = json.load(f)
    gcdc_desc = dataset_desc['gcdc']
    pers_desc = dataset_desc['persuasiveness']
    partisan_desc = dataset_desc['hyperpartisan']
    fake_news_desc = dataset_desc['fake_news']
    return fake_news_desc, gcdc_desc, partisan_desc, pers_desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_len", type=int, default=15, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=15, help="Max number of sentences in a doc")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    # parser.add_argument("--embed_size", type=int, default=768, help="Embedding size") # It is the output of BERT,
    # cannot be changed!
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training model in specific episode")
    parser.add_argument("--meta_lr", type=float, default=0.0001, help="Learning rate for updateing meta model")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--dataset_json", type=str, default='./dataset-paths.json',
                        help="JSON file containing the dataset paths")

    parser.add_argument("--meta_epochs", type=int, default=5, help="Number of meta epochs")
    parser.add_argument("--meta_batch", type=int, default=8, help="Size of meta batches")
    parser.add_argument("--n_episodes", type=int, default=7, help="Number of episodes per task")
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
