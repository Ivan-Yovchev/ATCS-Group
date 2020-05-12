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
import os
import time

def get_acc(preds, targets, loss):
    if type(loss) == nn.BCELoss:  # binary
        preds = preds > 0.5
    else:  # multiclass
        preds = preds.argmax(dim=-1)
    return torch.mean((preds == targets).float()).item()

def train_model(conv_model: nn.Module, sent_embedder: nn.Module,
                task_classifier: nn.Module, dataset: ParentDataset,
                loss: nn.Module, optim: torch.optim.Optimizer) -> float:
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
    conv_model.train()
    sent_embedder.train_bert()
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
            conv_model(
                sent_embedder(
                    document,
                    mask
                )
            )
        )

        # Compute loss
        grad = loss(out, label)

        # Backpropagate and upate weights
        grad.backward()
        optim.step()

        # Display results
        acc = get_acc(out, label, loss)
        avg_acc = (avg_acc * i + acc) / (i + 1)
        avg_loss = (avg_loss * i + grad.item()) / (i + 1)
        display_log.set_description_str(f"Batch {i:02d}:0 acc: {acc:.4f} loss: {grad.item():.4f}")

    display_log.close()
    return avg_acc, avg_loss


def eval_model(conv_model: nn.Module, doc_embedder: nn.Module, task_classifier: nn.Module,
               dataset: ParentDataset, loss: nn.Module) -> float:
    # Set all models to evaluation mode
    conv_model.eval()
    task_classifier.eval()
    doc_embedder.eval_bert()
    results = 0
    avg_loss = 0

    # Prevents the gradients from being computed
    with torch.no_grad():
        for i, (doc, mask, label) in tqdm(enumerate(dataset), total=len(dataset), position=0):
            # For each document compute the output
            out = torch.squeeze(
                task_classifier(
                    conv_model(
                        doc_embedder(
                            doc,
                            mask
                        )
                    )
                )
            )
            grad = loss(out.unsqueeze(0), label)

            avg_loss = (avg_loss * i + grad.item()) / (i + 1)
    return results / len(dataset), avg_loss


def main(args):
    writer = SummaryWriter(f'runs/{args.dataset_type}_{int(time.time())}')

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len, args.max_sent, args.batch_size, args.device)
    testset = get_dataset(args.dataset_type, args.test_path, bert_tokenizer, args.max_len, args.max_sent, 1, args.device)

    sent_embedder = None
    if args.doc_emb_type == "max_batcher":
        sent_embedder = BertManager(bert_model, args.max_len, args.device)
    # else if
    assert sent_embedder is not None

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
    elif args.dataset_type in ["hyperpartisan", "fake_news"]:
        criterion = nn.BCELoss()
        loss = lambda x,y: criterion(x.squeeze(1), y.float())

    assert loss is not None

    bert_model.to(args.device)
    conv_model.to(args.device)
    task_classifier.to(args.device)
    
    valid_acc, valid_loss = eval_model(conv_model, sent_embedder, task_classifier, testset, loss)
    print(f'Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
    best_acc = 0
    # optim = transformers.optimization.AdamW(list(model.parameters()) + list(bert_model.parameters()), args.lr)
    optim = transformers.optimization.AdamW(list(conv_model.parameters()) + list(task_classifier.parameters()), args.lr)
    # optim = transformers.optimization.AdamW(list(conv_model.parameters()), args.lr)
    lr_scheduler = ReduceLROnPlateau(optim, mode='max', patience=5)

    for epoch in range(args.n_epochs):
        
        if optim.defaults['lr'] < 1e-6: break

        train_acc, train_loss = train_model(conv_model, sent_embedder, task_classifier, dataset, loss, optim)

        valid_acc, valid_loss = eval_model(conv_model, sent_embedder, task_classifier, testset)
        print(f'Epoch {epoch + 1:02d}: train acc: {train_acc:.4f} train loss: {train_loss:.4f} valid acc: {valid_acc:.4f} valid loss: {valid_loss:.4f}')
        
        lr_scheduler.step(accuracy)

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
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
