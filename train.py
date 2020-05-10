import torch
import argparse

from torch import nn, flatten
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from doc_emb_models import *
from datasets import get_dataset, collate_pad_fn, ParentDataset
from cnn_model import CNNModel
from tqdm import tqdm
import transformers
import os


def _pad_sequence(t: torch.Tensor, to_seq_length=200):
    out_tensor = torch.zeros((t.size(0), to_seq_length, t.size(2)))


def train_model(conv_model: nn.Module, doc_embedder: nn.Module, task_classifier: nn.Module, dataset: ParentDataset,
                loss, optim):
    # important for BatchNorm layer
    conv_model.train()
    doc_embedder.train_bert()
    task_classifier.train()

    avg_loss = 0

    display_log = tqdm(dataset, total=0, position=1, bar_format='{desc}')

    for i, (document, mask, label) in tqdm(enumerate(dataset), total=len(dataset), position=0):
        # Reset gradients
        optim.zero_grad()

        # Compute output
        out = task_classifier(
            conv_model(
                doc_embedder(
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
        avg_loss = (avg_loss * i + grad.item()) / (i + 1)
        display_log.set_description_str(f"Current loss at {i:02d}: {grad.item():.4f}")

        # Do the job that the python garbage collector does?
        # del document, mask, label, grad

    display_log.close()
    return avg_loss


def eval_model(conv_model: nn.Module, doc_embedder: nn.Module, task_classifier: nn.Module, dataset: ParentDataset):
    conv_model.eval()
    task_classifier.eval()
    doc_embedder.eval_bert()
    results = 0
    with torch.no_grad():
        for doc, mask, label in tqdm(dataset):
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
            if not out.shape: # binary
                results += (out > 0.5).item() == label.item()
            else: # multiclass
                results += (out.argmax().item() == label.item())
    return results / len(dataset)


def main(args):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, 200, args.batch_size, args.device)
    testset = get_dataset(args.dataset_type, args.test_path, bert_tokenizer, 200, 1, args.device)

    doc_embedder = None
    if args.doc_emb_type == "max_batcher":
        doc_embedder = BertBatcher(bert_model, args.max_len, args.device)
    # else if
    assert doc_embedder is not None

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

    print(f'Accuracy at start: {eval_model(conv_model, doc_embedder, task_classifier,testset) :.4f}')
    best_acc = 0
    # optim = transformers.optimization.AdamW(list(model.parameters()) + list(bert_model.parameters()), args.lr)
    optim = transformers.optimization.AdamW(list(conv_model.parameters()) + list(task_classifier.parameters()), args.lr)
    # optim = transformers.optimization.AdamW(list(conv_model.parameters()), args.lr)

    for epoch in range(args.n_epochs):

        print("Epoch: ", epoch)

        avg_loss = train_model(conv_model, doc_embedder, task_classifier, dataset, loss, optim)
        print("Avg loss: ", avg_loss)

        accuracy = eval_model(conv_model, doc_embedder, task_classifier, testset)
        print(f'Accuracy at {epoch + 1:02d}: {accuracy:.4f}')

        if best_acc < accuracy:
            best_acc = accuracy

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
    parser.add_argument("--max_len", type=int, default=15, help="Max number of words contained in a sentence")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
