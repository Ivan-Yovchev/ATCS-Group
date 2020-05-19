import torch
import argparse
from torch import nn, flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from doc_emb_models import *
from datasets import *
from cnn_model import CNNModel
from tqdm import tqdm
import transformers
from common import Common
import os
from datetime import datetime


def get_acc(preds, targets, binary=False):
    if binary:  # binary
        preds = (preds > 0.5).to(torch.long)
    else:  # multiclass
        preds = preds.argmax(dim=-1)
    return torch.mean((preds == targets).float()).item()


def train_model(model: nn.Module, task_classifier: nn.Module, dataset: ParentDataset,
                loss: nn.Module, optim: torch.optim.Optimizer, binary: bool) -> float:
    """Performs an epoch of training on the provided model

    :param conv_model: the ConvNet model. Takes care of transforming the sentence embeddings into a document embedding
    :param sent_embedder: produces sentence embedding
    :param task_classifier: the task-specific classifier. The output must be consistent with the task loss.
    :param dataset: the dataset the models are trained on
    :param loss: the loss function
    :param optim: the optimizer the method should call after each batch
    :return: the average loss
    """

    # important for BatchNorm layer
    model.train()
    task_classifier.train()

    avg_acc = 0
    avg_loss = 0

    # display line
    display_log = tqdm(dataset, total=0, position=1, bar_format='{desc}')

    for i, (x, label) in tqdm(enumerate(dataset), total=len(dataset), position=0):
        # Reset gradients
        optim.zero_grad()

        # Compute output
        out = task_classifier(model(*x))
    
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


def eval_model(model: nn.Module, task_classifier: nn.Module,
               dataset: ParentDataset, loss: nn.Module, binary: bool) -> float:
    # Set all models to evaluation mode
    model.eval()
    task_classifier.eval()

    results = 0
    avg_loss = 0

    # Prevents the gradients from being computed
    with torch.no_grad():
        for i, (x, label) in tqdm(enumerate(dataset), total=len(dataset), position=0):
            # For each document compute the output
            out = task_classifier(
                model(
                    *x
                )
            )

            grad = loss(out, label)
            results += get_acc(out, label, binary)
            avg_loss = (avg_loss * i + grad.item()) / (i + 1)

    return results / len(dataset), avg_loss


def hyperpartisan_kfold_train(args):
    assert args.dataset_type in ["hyperpartisan"]
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    binary_classification, loss = loss_task_factory(args.dataset_type)

    kfold_dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len, args.max_sent,
                                args.batch_size, args.device, hyperpartisan_10fold=True)

    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    accuracy_list = []
    for fold, (trainset, testset) in enumerate(kfold_dataset):
        # logs
        writer = SummaryWriter(f'runs/{args.dataset_type}.{fold}_{time_log}')

        # reinitialize all the stuff
        task_classifier = task_classifier_factory(args)
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        sent_embedder = BertManager(bert_model, args.device)
        conv_model = CNNModel(args.embed_size, args.device, n_filters=args.n_filters, batch_norm_eval=True)
        conv_model.initialize_weights(nn.init.xavier_normal_)

        # construct common model
        model, trainset, testset = construct_common_model(args.finetune, conv_model, sent_embedder, trainset, testset)
        model.to(args.device)
        task_classifier.to(args.device)
        best_acc = 0
        optim = torch.optim.Adam(list(conv_model.parameters()) + list(task_classifier.parameters()), args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, mode='max', factor=0.8)
        # eval first time
        valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss=loss, binary=binary_classification)
        print(f'Fold {fold}. Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
        # start training
        for epoch in range(args.n_epochs):
            if optim.defaults['lr'] < 1e-6: break
            print("Epoch: ", epoch)
            train_acc, train_loss = train_model(model, task_classifier, trainset, loss, optim,
                                                binary_classification)
            print("Avg loss: ", train_loss)
            valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss,
                                               binary=binary_classification)
            # (model, task_classifier, testset, loss, binary=binary_classification
            print(f'Fold {fold}. Epoch {epoch:02d}: train acc: {train_acc:.4f}'
                  f' train loss: {train_loss:.4f} valid acc: {valid_acc:.4f}'
                  f' valid loss: {valid_loss:.4f}')
            lr_scheduler.step(valid_acc)

            writer.add_scalar('Train/accuracy', train_acc, epoch)
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Valid/accuracy', valid_acc, epoch)
            writer.add_scalar('Valid/loss', valid_loss, epoch)

            if best_acc < valid_acc:
                best_acc = valid_acc
                save_model(args.dataset_type, conv_model, bert_model, task_classifier, epoch, time_log, fold)

        accuracy_list.append(best_acc)
        del bert_model, task_classifier, conv_model
    average = sum(accuracy_list) / len(accuracy_list)
    print(f'average accuracy: {average}')
    return accuracy_list


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


def main(args):
    if args.kfold:
        hyperpartisan_kfold_train(args)
        return

    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(f'runs/{args.dataset_type}_{time_log}')

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len, args.max_sent,
                          args.batch_size if args.finetune else 1, args.device)
    testset = get_dataset(args.dataset_type, args.test_path, bert_tokenizer, args.max_len, args.max_sent,
                          args.batch_size if args.finetune else 1, args.device)

    sent_embedder = BertManager(bert_model, args.device)

    # loading task-specific classifier
    task_classifier = task_classifier_factory(args)
    conv_model = CNNModel(args.embed_size, args.device, n_filters=args.n_filters)

    binary_classification, loss = loss_task_factory(args.dataset_type)

    # construct common model
    model, dataset, testset = construct_common_model(args.finetune, conv_model, sent_embedder, dataset, testset)
    model.to(args.device)
    task_classifier.to(args.device)

    valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss, binary=binary_classification)
    print(f'Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
    best_acc = 0
    # optim = transformers.optimization.AdamW(list(model.parameters()) + list(bert_model.parameters()), args.lr)
    optim = torch.optim.Adam(list(model.parameters()) + list(task_classifier.parameters()), args.lr, weight_decay=0.02)
    # optim = transformers.optimization.AdamW(list(conv_model.parameters()), args.lr)

    lr_scheduler = ReduceLROnPlateau(optim, mode='max', patience=5, factor=0.8)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.8)

    for epoch in range(args.n_epochs):

        if optim.defaults['lr'] < 1e-6: break
        train_acc, train_loss = train_model(model, task_classifier, dataset, loss, optim, binary=binary_classification)
        valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss, binary=binary_classification)
        print(f'Epoch {epoch:02d}: train acc: {train_acc:.4f}'
              f' train loss: {train_loss:.4f} valid acc: {valid_acc:.4f}'
              f' valid loss: {valid_loss:.4f}')

        lr_scheduler.step(valid_acc)

        writer.add_scalar('Train/accuracy', train_acc, epoch)
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Valid/accuracy', valid_acc, epoch)
        writer.add_scalar('Valid/loss', valid_loss, epoch)

        if best_acc < valid_acc:
            best_acc = valid_acc
            save_model(args.dataset_type, conv_model, bert_model, task_classifier, epoch, time_log)


def construct_common_model(finetune, conv_model, sent_embedder, dataset, testset):
    if finetune:
        model = Common(conv_model, encoder=sent_embedder)
    else:
        model = Common(conv_model)
        dataset = BertPreprocessor(dataset, sent_embedder, conv_model.get_max_kernel(), batch_size=args.batch_size)
        testset = BertPreprocessor(testset, sent_embedder, conv_model.get_max_kernel(), batch_size=args.batch_size)
    return model, dataset, testset


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
    if args.dataset_type == "gcdc":
        task_classifier = nn.Linear(5 * args.n_filters, 3)
    elif args.dataset_type in ["hyperpartisan", "fake_news"]:
        task_classifier = nn.Sequential(nn.Linear(5 * args.n_filters, 1), nn.Sigmoid())
    elif args.dataset_type == "persuasiveness":
        task_classifier = nn.Linear(5 * args.n_filters, 6)
    assert task_classifier is not None, 'task not recognized'
    return task_classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--train_path", type=str, default="data/GCDC/Clinton_train.csv", help="Path to training data")
    parser.add_argument("--test_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to testing data")
    parser.add_argument("--max_len", type=int, default=50, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=100, help="Max number of sentences per document")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--kfold", type=lambda x: x.lower() == "true", default=False,
                        help="10fold for hyperpartisan dataset. test_path value will be ignored")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False,
                        help="Set to true to fine tune bert")
    args = parser.parse_args()
    args.embed_size = 768
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
