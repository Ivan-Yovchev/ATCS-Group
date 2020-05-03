import torch
import argparse

from torch import nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from datasets import get_dataset, collate_pad_fn
from cnn_model import CNNModel
from tqdm import tqdm


def train_model(conv_model, bert_model: BertModel, dataset, loss, optim, device, threshold=False):
    # important for BatchNorm layer
    conv_model.train()

    avg_loss = 0
    # for i in range(len(dataloader)):
    #     batch = dataloader[i]
    for i, (document, mask, label) in tqdm(enumerate(dataset)):
        optim.zero_grad()
        label = label.to(device, torch.float32)
        mask = mask.to(device)
        # x_ = [bert_model(x.to(device).unsqueeze(0))[0].permute(0, 2, 1) for x in document]
        print(document.shape, mask.shape)
        x = bert_model(document.to(device), attention_mask=mask)
        print(x[0].shape)
        x = torch.sum(x[0]*mask.unsqueeze(-1), dim=1)/mask.sum(dim=1).unsqueeze(-1)
        x = x.unsqueeze(0).permute(0, 2, 1)
        print(x.shape)
        out = conv_model(x)

        # grad = loss(out > 0.5 if threshold else out, labels)
        grad = loss(out.squeeze(), label)

        avg_loss = (avg_loss * i + grad.item()) / (i + 1)

        grad.backward()
        optim.step()

    return avg_loss


def eval_model(model, dataloader, device, threshhold=False):
    model.eval()

    for i in range(len(dataloader)):
        out = model(batch[0].permute(0, 2, 1).to(device))

        # comparison
        correct += ().sum().item()

    return correct / (len(dataloader) * dataloader.batch_size)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_length=200, pad_token='<PAD>',
                                                   pad_to_max_length=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, args.max_len)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad_fn)

    classifier = None
    if args.dataset_type == "gcdc":
        classifier = nn.Sequential(nn.Linear(5 * args.n_filters, 1), nn.Sigmoid())
    # else if
    assert classifier is not None

    model = CNNModel(args.embed_size, args.max_len, classifier, device, n_filters=args.n_filters)
    loss = None
    if args.dataset_type == "gcdc":
        loss = nn.BCELoss()
    # else if

    assert loss is not None

    optim = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.n_epochs):
        print("Epoch: ", epoch)

        avg_loss = train_model(model, bert_model, dataset, loss, optim, device=device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
