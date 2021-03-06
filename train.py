import torch
import argparse
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple, Mapping
import numpy as np
from datetime import datetime
from doc_emb_models import *
from datasets import *
from cnn_model import CNNModel
from common import Common
from utils import get_acc, AccumulatorF1, load_model, save_model, get_summary_writer

# torch.manual_seed(42)
# np.random.seed(42)


def train_model(model: nn.Module, task_classifier: nn.Module, dataset: ParentDataset,
                loss: nn.Module, optim: torch.optim.Optimizer, binary: bool, disp_tqdm: bool = True) -> Tuple[
    float, float]:
    """Performs an epoch of training on the provided model

    :param conv_model: the ConvNet model. Takes care of transforming the sentence embeddings into a document embedding
    :param sent_embedder: produces sentence embedding
    :param task_classifier: the task-specific classifier. The output must be consistent with the task loss.
    :param dataset: the dataset the models are trained on
    :param loss: the loss function
    :param optim: the optimizer the method should call after each batch
    :param binary: if the task is binary or not
    :param disp_tqdm: whether to display the progress bar or not (default: True).
    :return: tuple containing average accuracy and average loss
    """

    # important for BatchNorm layer
    model.train()
    task_classifier.train()

    avg_acc = 0
    avg_loss = 0

    # display line
    display_log = tqdm(dataset, total=0, position=1, bar_format='{desc}', disable=not disp_tqdm)

    for i, (x, label) in tqdm(enumerate(dataset), total=len(dataset), position=0, disable=not disp_tqdm):
        # Reset gradients
        optim.zero_grad()

        # Compute output
        # x will be unpacked for compatibility with the finetuning mode
        out = task_classifier(model(x))

        # Compute loss
        grad = loss(out, label)

        # Backpropagate and update weights
        grad.backward()
        optim.step()

        # Display results
        acc = get_acc(out, label, binary)
        avg_acc = (avg_acc * i + acc) / (i + 1)
        avg_loss = (avg_loss * i + grad.item()) / (i + 1)
        display_log.set_description_str(f"Batch {i:02d}:0 acc: {acc:.4f} loss: {grad.item():.4f}")

    display_log.close()
    return avg_acc, avg_loss


def eval_model(model: nn.Module, task_classifier: nn.Module, dataset: ParentDataset, loss: nn.Module, binary: bool,
               disp_tqdm: bool = True, special_binary: bool = False) -> Tuple[float, float, Mapping]:
    # Set all models to evaluation mode
    model.eval()
    task_classifier.eval()

    results = 0
    avg_loss = 0

    f1_acc = AccumulatorF1() if binary or special_binary else None

    # Prevents the gradients from being computed
    with torch.no_grad():
        for i, (x, label) in tqdm(enumerate(dataset), total=len(dataset), position=0, disable=not disp_tqdm):
            # For each document compute the output
            out = task_classifier(model(x))
            grad = loss(out, label)
            results += get_acc(out, label, binary)
            if binary or special_binary:
                f1_acc.add(out, label)

            avg_loss = (avg_loss * i + grad.item()) / (i + 1)

    f1_stats = f1_acc.reduce() if binary or special_binary else None

    return results / len(dataset), avg_loss, f1_stats


def eval_test(model: nn.Module, task_classifier: nn.Module,
              dataset: ParentDataset, loss: nn.Module, binary: bool, disp_tqdm: bool = True):
    accs = []
    losses = []
    if binary:
        precisions = []
        recalls = []
        f1s = []
    for seed in np.random.randint(0, 100, size=10):
        torch.manual_seed(seed)
        cur_acc, cur_loss, f1stats = eval_model(model, task_classifier, dataset, loss, binary, disp_tqdm)
        accs.append(cur_acc)
        losses.append(cur_loss)
        if binary:
            precisions.append(f1stats[0])
            recalls.append(f1stats[1])
            f1s.append(f1stats[2])

    if binary:
        f1stats = (np.mean(precisions), np.std(precisions)), (np.mean(recalls), np.std(recalls)), (np.mean(f1s), np.std(f1s))
    else:
        f1stats = None
    return (np.mean(accs), np.std(accs)), (np.mean(losses), np.std(losses)), f1stats


def hyperpartisan_kfold_train(args):
    assert args.dataset_type in ["hyperpartisan"]
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    binary_classification, loss = loss_task_factory(args.dataset_type)

    kfold_dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len, args.max_sent,
                                1, args.device, hyperpartisan_10fold=True)

    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    accuracy_list = []
    for fold, (trainset, testset) in enumerate(kfold_dataset):
        # logs
        writer = SummaryWriter(f'runs/{args.dataset_type}.{fold}_{time_log}')

        # reinitialize all the stuff
        task_classifier = task_classifier_factory(args)
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        sent_embedder = BertManager(bert_model, args.device)
        conv_model = CNNModel(args.embed_size, args.device, n_filters=args.n_filters, filter_sizes=args.kernels)
        conv_model.initialize_weights(nn.init.xavier_normal_)

        # construct common model
        model = construct_common_model(args.finetune, conv_model, sent_embedder)
        trainset = BertPreprocessor(trainset, sent_embedder, conv_model.get_max_kernel(), batch_size=args.batch_size)
        testset = BertPreprocessor(testset, sent_embedder, conv_model.get_max_kernel(), batch_size=args.batch_size)

        model.to(args.device)
        task_classifier.to(args.device)

        best_acc = 0
        optim = torch.optim.Adam(list(conv_model.parameters()) + list(task_classifier.parameters()), args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, mode='max', factor=0.8)
        # eval first time
        valid_acc, valid_loss, f1stats = eval_model(model, task_classifier, testset, loss=loss,
                                                    binary=binary_classification)
        print(f'Fold {fold}. Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
        # start training
        for epoch in range(args.n_epochs):
            if optim.defaults['lr'] < 1e-6: break
            print("Epoch: ", epoch)
            train_acc, train_loss = train_model(model, task_classifier, trainset, loss, optim,
                                                binary_classification)
            print("Avg loss: ", train_loss)
            valid_acc, valid_loss, f1stats = eval_model(model, task_classifier, testset, loss,
                                                        binary=binary_classification)
            print(f'Fold {fold}. Epoch {epoch:02d}: train acc: {train_acc:.4f}'
                  f' train loss: {train_loss:.4f} valid acc: {valid_acc:.4f}'
                  f' valid loss: {valid_loss:.4f}')
            lr_scheduler.step(valid_acc)

            writer.add_scalar('Train/accuracy', train_acc, epoch)
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Valid/accuracy', valid_acc, epoch)
            writer.add_scalar('Valid/loss', valid_loss, epoch)

            if binary_classification:
                writer.add_scalar('Valid/precision', f1stats[0], epoch)
                writer.add_scalar('Valid/recall', f1stats[1], epoch)
                writer.add_scalar('Valid/f1', f1stats[2], epoch)

            if best_acc < valid_acc:
                best_acc = valid_acc
                save_model(args.dataset_type, conv_model, bert_model, task_classifier, epoch, time_log, fold)

        accuracy_list.append(best_acc)
        del bert_model, task_classifier, conv_model
    average = sum(accuracy_list) / len(accuracy_list)
    print(f'average accuracy: {average}')
    return accuracy_list


def construct_common_model(finetune, conv_model, sent_embedder):
    if finetune:
        model = Common(conv_model, 
                       n_filters=conv_model.get_n_blocks() * args.n_filters,
                       encoder=sent_embedder)
        
    else:
        model = Common(conv_model, 
                       n_filters=conv_model.get_n_blocks() * args.n_filters)
    return model


def loss_task_factory(dataset_type):
    loss = None
    if dataset_type in ["gcdc", "persuasiveness"]:
        loss = nn.CrossEntropyLoss()
        binary_classification = False
    elif dataset_type in ["hyperpartisan", "fake_news"]:
        binary_classification = True
        criterion = nn.BCELoss()
        loss = lambda x, y: criterion(x.squeeze(1), y.float())
    assert loss is not None, 'task not recognized'
    return binary_classification, loss


def task_classifier_factory(args):
    task_classifier = None
    input_dim = len(args.kernels) * args.n_filters
    if args.dataset_type == "gcdc":
        task_classifier = nn.Linear(input_dim, 3)
    elif args.dataset_type in ["hyperpartisan", "fake_news"]:
        task_classifier = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
    elif args.dataset_type == "persuasiveness":
        task_classifier = nn.Linear(input_dim, 6)
    assert task_classifier is not None, 'task not recognized'
    return task_classifier


def get_datasets(args, bert_tokenizer, sent_embedder, max_kernel):
    trainset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len, args.max_sent,
                           args.batch_size if args.finetune else 1, args.device)
    validset = get_dataset(args.dataset_type, args.valid_path, bert_tokenizer, args.max_len, args.max_sent,
                           args.batch_size if args.finetune else 1, args.device)
    testset = get_dataset(args.dataset_type, args.test_path, bert_tokenizer, args.max_len, args.max_sent,
                          args.batch_size if args.finetune else 1, args.device)
    if not args.finetune:
        trainset = BertPreprocessor(trainset, sent_embedder, max_kernel, batch_size=args.batch_size)
        validset = BertPreprocessor(validset, sent_embedder, max_kernel, batch_size=args.batch_size)
        testset = BertPreprocessor(testset, sent_embedder, max_kernel, batch_size=args.batch_size)

    return trainset, validset, testset


def main(args):
    if args.kfold:
        hyperpartisan_kfold_train(args)
        return

    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = get_summary_writer(args, time_log)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    sent_embedder = BertManager(bert_model, args.device)

    # loading task-specific classifier
    task_classifier = task_classifier_factory(args)
    conv_model = CNNModel(args.embed_size, args.device, n_filters=args.n_filters, filter_sizes=args.kernels)

    binary_classification, loss = loss_task_factory(args.dataset_type)

    # construct common model
    model = construct_common_model(args.finetune, conv_model, sent_embedder)
    model.to(args.device)
    task_classifier.to(args.device)

    trainset, validset, testset = get_datasets(args, bert_tokenizer, sent_embedder, conv_model.get_max_kernel())

    valid_acc, valid_loss, f1stats = eval_model(model, task_classifier, validset, loss, binary=binary_classification)
    print(f'Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
    best_acc = 0
    optim = torch.optim.Adam(list(model.parameters()) + list(task_classifier.parameters()), args.lr)

    lr_scheduler = ReduceLROnPlateau(optim, mode='max', patience=5, factor=0.8)

    for epoch in range(args.n_epochs):

        if optim.defaults['lr'] < 1e-6: break
        train_acc, train_loss = train_model(model, task_classifier, trainset, loss, optim, binary=binary_classification)
        valid_acc, valid_loss, f1stats = eval_model(model, task_classifier, validset, loss,
                                                    binary=binary_classification)
        print(f'Epoch {epoch:02d}: train acc: {train_acc:.4f}'
              f' train loss: {train_loss:.4f} valid acc: {valid_acc:.4f}'
              f' valid loss: {valid_loss:.4f}')

        lr_scheduler.step(valid_acc)

        writer.add_scalar('Train/accuracy', train_acc, epoch)
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Valid/accuracy', valid_acc, epoch)
        writer.add_scalar('Valid/loss', valid_loss, epoch)
        if binary_classification:
            writer.add_scalar('Valid/precision', f1stats[0], epoch)
            writer.add_scalar('Valid/recall', f1stats[1], epoch)
            writer.add_scalar('Valid/f1', f1stats[2], epoch)

        if best_acc < valid_acc:
            best_acc = valid_acc
            save_model(args.dataset_type, conv_model, bert_model, task_classifier, epoch, time_log)

    best_model, best_task_classifier = load_model(f"models/{args.dataset_type}.{time_log}.pt",
                                                  conv_model, task_classifier, sent_embedder, args.finetune)
    (test_acc, test_acc_std), (test_loss, test_loss_std) = eval_test(best_model, best_task_classifier, testset,
                                                                     loss, binary=binary_classification,
                                                                     disp_tqdm=False)

    writer.add_hparams({
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'max_sent': args.max_sent,
        'n_filters': args.n_filters,
        'kernels': str(args.kernels),
        'lr': args.lr,
        'test_acc': test_acc,
        'test_acc_std': test_acc_std,
        'test_loss': test_loss,
        'test_loss_std': test_loss_std,
    }, {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--train_path", type=str, default="data/GCDC/Clinton_train.csv", help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="data/GCDC/Enron_test.csv", help="Path to validation data")
    parser.add_argument("--test_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to testing data")
    parser.add_argument("--max_len", type=int, default=50, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=50, help="Max number of sentences per document")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--kfold", type=lambda x: x.lower() == "true", default=False,
                        help="10fold for hyperpartisan dataset. test_path value will be ignored")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False,
                        help="Set to true to fine tune bert")
    parser.add_argument("--kernels", type=lambda x: [int(i) for i in x.split(',')], default="2,4,6",
                        help="Kernel sizes per cnn block")
    args = parser.parse_args()
    args.embed_size = 768
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
