import torch
import argparse

from torch import nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from datasets import get_dataset, collate_pad_fn
from cnn_model import CNNModel

def train_model(model, dataloader, loss, optim, device, threshhold = False):

    # important for BatchNorm layer
    model.train()

    avg_loss = 0
    for i in range(len(dataloader)):
        batch = dataloader[i]

        optim.zero_grad()

        out = model(batch[0].permute(0, 2, 1).to(device))

        grad = loss(out > 0.5 if threshhold else out, batch[1].to(device))

        avg_loss = (avg_loss * i + grad.item()) / (i + 1)

        grad.backward()
        optim.step()

    return avg_loss

def eval_model(model, dataloader, device, threshhold = False):

    model.eval()

    for i in range(len(dataloader)):
        out = model(batch[0].permute(0, 2, 1).to(device))

        # comparison
        correct += ().sum().item()

    return correct / (len(dataloader) * dataloader.batch_size)

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad_fn)

    classifier = None
    if args.dataset_type == "gcdc":
        classifier = nn.Sequential(nn.Linear(5*args.n_filters, 1), nn.Sigmoid())
    # else if

    assert (classifier != None)

    model = CNNModel(args.embed_size, args.max_len, classifier, device, n_filters=args.n_filters)

    loss = None
    if args.dataset_type == "gcdc":
        loss =  nn.BCELoss()
    # else if

    assert (loss != None)

    optim = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in args.n_epochs:

        print("Epoch: ", epoch)

        avg_loss = train_model(model, dataloader, loss, optim)

        print("Avg loss: ", avg_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--train_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to training data")
    parser.add_argument("--test_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to testing data")
    parser.add_argument("--max_len", type=int, default=300, help="Max number of words contained in a sentence")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")

    args = parser.parse_args()

    main(args)
