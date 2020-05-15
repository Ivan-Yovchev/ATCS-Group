import torch
import argparse
from torch import nn, flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from doc_emb_models import *
from datasets import get_dataset, collate_pad_fn, ParentDataset
from cnn_model import CNNModel
from tqdm import tqdm
import transformers
from common import Common
import os
import time


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

    for i, (document, mask, label) in tqdm(enumerate(dataset), total=len(dataset), position=0):
        # Reset gradients
        optim.zero_grad()

        # Compute output
        out = task_classifier(
            model(
                document,
                mask
            )
        )

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
        for i, (doc, mask, label) in tqdm(enumerate(dataset), total=len(dataset), position=0):
            # For each document compute the output
            out = task_classifier(
                model(
                    doc,
                    mask
                )
            )

            grad = loss(out, label)
            results += get_acc(out, label, binary)
            avg_loss = (avg_loss * i + grad.item()) / (i + 1)

    return results / len(dataset), avg_loss


def hyperpartisan_kfold_train(args):
    assert args.dataset_type in ["hyperpartisan"]
    binary_classification = True

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    criterion = nn.BCELoss()
    loss = lambda x, y: criterion(x.squeeze(1), y.float())

    kfold_dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len, args.max_sent,
                                args.batch_size, args.device, hyperpartisan_10fold=True)

    log_time = int(time.time())
    accuracy_list = []
    for fold, (trainset, testset) in enumerate(kfold_dataset):
        # logs
        writer = SummaryWriter(f'runs/{args.dataset_type}.{fold}_{log_time}')

        # reinitialize all the stuff
        task_classifier = nn.Sequential(nn.Linear(5 * args.n_filters, 1), nn.Sigmoid())
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        sent_embedder = BertManager(bert_model, args.max_len, args.device)
        conv_model = CNNModel(args.embed_size, args.max_len, args.device, n_filters=args.n_filters)
        bert_model.to(args.device)
        conv_model.to(args.device)
        task_classifier.to(args.device)
        best_acc = 0
        optim = torch.optim.Adam(list(conv_model.parameters()) + list(task_classifier.parameters()), args.lr,
                                 weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, mode='max', factor=0.8)

        # eval first time
        valid_acc, valid_loss = eval_model(conv_model, sent_embedder, task_classifier, testset, loss=loss,
                                           binary=binary_classification)
        print(f'Fold {fold}. Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
        # start training
        for epoch in range(args.n_epochs):
            if optim.defaults['lr'] < 1e-6: break
            print("Epoch: ", epoch)
            train_acc, train_loss = train_model(conv_model, sent_embedder, task_classifier, trainset, loss, optim,
                                                binary_classification)
            print("Avg loss: ", train_loss)
            valid_acc, valid_loss = eval_model(conv_model, sent_embedder, task_classifier, testset, loss=loss,
                                               binary=binary_classification)
            print(f'Fold {fold}. At {epoch:02d}: acc: {valid_acc:.4f}, loss: {valid_loss}')

            lr_scheduler.step(valid_acc)

            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('valid_acc', valid_acc, epoch)
            writer.add_scalar('valid_loss', valid_loss, epoch)

            if best_acc < valid_acc:
                best_acc = valid_acc

                with open(os.path.join('models', args.dataset_type + f".{fold}_{log_time}.pt"), 'wb') as f:
                    torch.save({
                        'cnn_model': conv_model.state_dict(),
                        'bert_model': bert_model.state_dict(),
                        'task_classifier': task_classifier.state_dict(),
                        'epoch': epoch
                    }, f)
        accuracy_list.append(best_acc)
        del bert_model, task_classifier, conv_model
    average = sum(accuracy_list) / len(accuracy_list)
    print(f'average accuracy: {average}')
    return accuracy_list


def main(args):
    if args.kfold:
        hyperpartisan_kfold_train(args)
        return

    writer = SummaryWriter(f'runs/{args.dataset_type}_{int(time.time())}')

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len, args.max_sent,
                          args.batch_size, args.device)
    testset = get_dataset(args.dataset_type, args.test_path, bert_tokenizer, args.max_len, args.max_sent,
                          args.batch_size, args.device)

    sent_embedder = None
    if args.doc_emb_type == "max_batcher":
        sent_embedder = BertManager(bert_model, args.max_len, args.device)
    # else if
    assert sent_embedder is not None

    # loading task-specific classifier
    task_classifier = None
    if args.dataset_type == "gcdc":
        task_classifier = nn.Linear(5 * args.n_filters, 3)
    elif args.dataset_type in ["hyperpartisan", "fake_news"]:
        task_classifier = nn.Sequential(nn.Linear(5 * args.n_filters, 1), nn.Sigmoid())
    elif args.dataset_type == "persuasiveness":
        task_classifier = nn.Linear(5 * args.n_filters, 6)
    assert task_classifier is not None

    conv_model = CNNModel(args.embed_size, args.max_len, args.device, n_filters=args.n_filters)

    loss = None
    if args.dataset_type in ["gcdc", "persuasiveness"]:
        loss = nn.CrossEntropyLoss()
        binary_classification = False
    elif args.dataset_type in ["hyperpartisan", "fake_news"]:
        binary_classification = True
        criterion = nn.BCELoss()
        loss = lambda x, y: criterion(x.squeeze(1), y.float())

    assert loss is not None

    # construct common model
    model = Common(conv_model, encoder = sent_embedder)

    bert_model.to(args.device)
    conv_model.to(args.device)
    task_classifier.to(args.device)

    valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss, binary=binary_classification)
    print(f'Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
    best_acc = 0
    # optim = transformers.optimization.AdamW(list(model.parameters()) + list(bert_model.parameters()), args.lr)
    optim = torch.optim.Adam(list(model.parameters()) + list(task_classifier.parameters()), args.lr)
    # optim = transformers.optimization.AdamW(list(conv_model.parameters()), args.lr)

    # lr_scheduler = ReduceLROnPlateau(optim, mode='max', patience=5, factor=0.8)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.8)

    for epoch in range(args.n_epochs):

        if optim.defaults['lr'] < 1e-6: break

        train_acc, train_loss = train_model(model, task_classifier, dataset, loss, optim, binary=binary_classification)

        valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss, binary=binary_classification)
        print(
            f'Epoch {epoch + 1:02d}: train acc: {train_acc:.4f} train loss: {train_loss:.4f} valid acc: {valid_acc:.4f} valid loss: {valid_loss:.4f}')

        lr_scheduler.step(valid_acc)

        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_acc', valid_acc, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)

        if best_acc < valid_acc:
            best_acc = valid_acc

            with open(os.path.join('models', args.dataset_type + ".pt"), 'wb') as f:
                torch.save({
                    'cnn_model': conv_model.state_dict(),
                    'bert_model': bert_model.state_dict(),
                    'task_classifier': task_classifier.state_dict(),
                    'epoch': epoch
                }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--train_path", type=str, default="data/GCDC/Clinton_train.csv", help="Path to training data")
    parser.add_argument("--test_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to testing data")
    parser.add_argument("--max_len", type=int, default=50, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=100, help="Max number of sentences per document")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--kfold", type=bool, default=False,
                        help="10fold for hyperpartisan dataset. test_path value will be ignored")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
