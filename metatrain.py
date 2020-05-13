import torch
import transformers

from tqdm import tqdm
from torch import nn
from transformers import BertModel, BertTokenizer
from datasets import EpisodeMaker
from doc_emb_models import *
from cnn_model import CNNModel

from train import train_model, eval_model

from copy import deepcopy
import argparse

class Common(nn.Module):

    def __init__(self, encoder, cnn, n_filters):
        super().__init__()

        self.encoder = encoder
        self.cnn = cnn

        # Proto learner
        self.n_filters = n_filters

    def forward(self, *args):
        return self.cnn(self.encoder(*args))

    def get_outputlayer(self, S, n_classes):

        C = torch.zeros(n_classes, self.n_filters)
        l2i = {}

        for doc, mask, label in S:

            # Label to index
            l2i[label.item()] = l2i.get(label.item(), len(l2i))
            idx = l2i[label.item()]

            # Accumulate latent vectors
            C[idx] += self(doc, mask).detach().squeeze().cpu()

        # Assume equal number of examples for each class
        samples_per_class = len(S) / len(l2i)
        C /= samples_per_class

        # Replace W and b in linear layer
        linear = nn.Linear(self.n_filters, n_classes)
        linear.weight = nn.Parameter(2*C)
        linear.bias = nn.Parameter(-torch.diag(C @ C.T))

        linear.to(self.cnn.device)
        return linear, C.detach() # C should already be detached

class Task:
    '''
    contains all data needed with regards to a specific task
    '''
    def __init__(self, get_episode, loss, n_classes):

        self.get_episode = get_episode # should be a lambda
        self.n_classes = n_classes
        self.loss = loss

def train_support(model: nn.Module, task: Task, init_optim):

    # Get episode
    ep = task.get_episode()

    # Construct output layer
    task_classifier, protos = model.get_outputlayer(ep["support_set"], task.n_classes)

    # Construct copy of doc_embedder to simulate training on current task
    model_cp = deepcopy(model)

    # Initialize optimizer for copy
    optim = init_optim(model_cp.parameters())

    train_model(model_cp.cnn, model_cp.encoder, task_classifier, ep["support_set"], task.loss, optim, False)

    # Step 6 from FO-Proto MAML pdf
    task_classifier.weight = nn.Parameter(protos + (task_classifier.weight - protos).detach())        

    # Get gradients for main (original) model update

    # Reset gradients
    optim.zero_grad()

    return model_cp, ep, task_classifier

def run_task_batch(model: nn.Module, tasks, init_optim, lr):

    # Store gradients for each simulated model (trained on each task in the given batch)
    meta_grads = {}

    for task in tasks:

        model_cp, ep, task_classifier = train_support(model, task, init_optim)

        # Train on task episode
        for doc, mask, labels in ep["query_set"]: # ISSUE this should be a different set

            # Compute output
            out = task_classifier(
                model_cp(
                    doc,
                    mask
                )
            )

            # Compute loss
            err = task.loss(out, labels)

            # Backpropagate and accumulate gradients (per batch)
            err.backward()

        # Accumulate gradients (per task)
        for par_name, par in dict(list(model_cp.named_parameters())).items():

            grad = par.grad/len(ep["query_set"])

            if par_name not in meta_grads:
                meta_grads[par_name] = grad
            else:
                meta_grads[par_name] += grad

    # Apply gradients
    for par_name, par in dict(list(model.named_parameters())).items():
        par = par - lr*meta_grads[par_name]

    model.zero_grad()

def meta_valid(model: nn.Module, task: Task, init_optim):
    model_cp, ep, task_classifier = train_support(model, task, init_optim)
    return eval_model(model_cp.cnn, model_cp.encoder, task_classifier, ep["query_set"], task.loss, False)

def main(args):

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Define tasks episodes
    gcdc_desc = {
        "name": "gcdc",
        "train": "./data/GCDC/",
        "test": "./data/GCDC"
    }

    pers_desc = {
        "name": "persuasiveness",
        "train": "./data/DebatePersuasiveness/persuasiveness_dataset-train.json",
        "test": "./data/DebatePersuasiveness/persuasiveness_dataset-test.json"
    }

    ep_maker = EpisodeMaker(bert_tokenizer, args.max_len, args.max_sent, args.device, [gcdc_desc, pers_desc])

    # Define tasks
    tasks = [
        Task(
            lambda : ep_maker.get_episode('gcdc', 1), # need to change this so that '1' changes over time
            nn.CrossEntropyLoss(),
            3
        ),
        Task(
            lambda : ep_maker.get_episode('persuasiveness', 1), # need to change this so that '1' changes over time
            nn.CrossEntropyLoss(),
            6
        )
    ]

    # Init Bert layer
    sent_embedder = BertManager(bert_model, args.max_len, args.device)

    # Init Conv layer
    conv_model = CNNModel(args.embed_size, args.max_len, args.device, n_filters=args.n_filters)

    # Build unified model
    model = Common(
        sent_embedder,
        conv_model,
        5*args.n_filters
    )

    bert_model.to(args.device)
    conv_model.to(args.device)

    # Define optimizer constructor
    init_optim = lambda pars : transformers.optimization.AdamW(pars, args.lr)

    # meta train
    display_log = tqdm(range(args.meta_epochs), total=0, position=9, bar_format='{desc}')
    for i in tqdm(range(args.meta_epochs), desc="Meta-epochs", total=args.meta_epochs, position=8):
        run_task_batch(model, tasks, init_optim, args.lr)

        # Meta Validation
        acc, loss = meta_valid(model, tasks[0], init_optim)
        display_log.set_description_str(f"Meta-valid {i:02d} acc: {acc:.4f} loss: {loss:.4f}")
    display_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--train_path", type=str, default="data/GCDC/Clinton_train.csv", help="Path to training data")
    parser.add_argument("--test_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to testing data")
    parser.add_argument("--max_len", type=int, default=15, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=15, help="Max number of sentences in a doc")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")

    parser.add_argument("--meta_epochs", type=int, default=5, help="Number of meta epochs")

    args = parser.parse_args()
    # args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cpu")
    main(args)
