import torch
import argparse

from torch import nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from datasets import get_dataset, collate_pad_fn
from cnn_model import CNNModel
from tqdm import tqdm
import transformers
import os


def _pad_sequence(t: torch.Tensor, to_seq_length=200):
    out_tensor = torch.zeros((t.size(0), to_seq_length, t.size(2)))


def train_model(conv_model, bert_model: BertModel, dataset, loss, optim, device, threshold=False):
    # important for BatchNorm layer
    conv_model.train()
    bert_model.train()

    avg_loss = 0
    # for i in range(len(dataloader)):
    #     batch = dataloader[i]
    display_log = tqdm(total=0, position=2, bar_format='{desc}')
    shuffled_order = list(range(len(dataset)))
    for i, idx in tqdm(enumerate(shuffled_order), total=len(dataset), position=1):
        document, mask, label = dataset[idx]
        optim.zero_grad()
        label = label.to(device, torch.float32)
        mask = mask.to(device)
        x = bert_model(document.to(device), attention_mask=mask)
        x = torch.sum(x[0] * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1).unsqueeze(-1)
        x = x.unsqueeze(0).permute(0, 2, 1)
        x = torch.nn.functional.pad(x, (0, args.max_len - x.size(2)))
        out = conv_model(x)

        # grad = loss(out > 0.5 if threshold else out, labels)
        grad = loss(out.squeeze(), label)

        avg_loss = (avg_loss * i + grad.item()) / (i + 1)
        display_log.set_description_str(f"Current loss at {i:02d}: {grad.item():.4f}")
        grad.backward()
        optim.step()
        del document, mask, label, grad

    display_log.close()
    return avg_loss


def eval_model(bert_model, conv_model, dataset, device):
    conv_model.eval()
    bert_model.eval()
    results = 0
    with torch.no_grad():
        for doc, mask, label in tqdm(dataset):
            mask = mask.to(device)
            x = bert_model(doc.to(device), attention_mask=mask)
            x = torch.sum(x[0] * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1).unsqueeze(-1)
            x = x.unsqueeze(0).permute(0, 2, 1)
            x = torch.nn.functional.pad(x, (0, args.max_len - x.size(2)))
            out = conv_model(x).item()
            results += ((out >= 0.5) == label.item())

    return results / len(dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    dataset = get_dataset(args.dataset_type, args.train_path, bert_tokenizer, 200)
    testset = get_dataset(args.dataset_type, args.test_path, bert_tokenizer, 200)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad_fn)

    classifier = None
    if args.dataset_type == "gcdc":
        classifier = nn.Sequential(nn.Linear(5 * args.n_filters, 1), nn.Sigmoid())
    # else if
    assert classifier is not None

    conv_model = CNNModel(args.embed_size, args.max_len, classifier, device, n_filters=args.n_filters)
    loss = None
    if args.dataset_type == "gcdc":
        loss = nn.BCELoss()
    # else if

    assert loss is not None

    print(f'Accuracy at start: {eval_model(bert_model, conv_model, testset, device):.4f}')
    best_acc = 0
    # optim = transformers.optimization.AdamW(list(model.parameters()) + list(bert_model.parameters()), args.lr)
    optim = transformers.optimization.AdamW(list(conv_model.parameters()), args.lr)

    for epoch in range(args.n_epochs):
        print("Epoch: ", epoch)
        avg_loss = train_model(conv_model, bert_model, dataset, loss, optim, device=device)
        print("Avg loss: ", avg_loss)
        accuracy = eval_model(bert_model, conv_model, testset, device)
        print(f'Accuracy at {epoch + 1:02d}: {accuracy:.4f}')
        if best_acc < accuracy:
            best_acc = accuracy
            with open(os.path.join('models', args.dataset_type + ".pt"), 'wb') as f:
                torch.save({
                    'cnn_model': conv_model.state_dict(),
                    'bert_model': bert_model.state_dict(),
                    'epoch': epoch + 1
                }, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--train_path", type=str, default="data/GCDC/Clinton_train.csv", help="Path to training data")
    parser.add_argument("--test_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to testing data")
    parser.add_argument("--max_len", type=int, default=15, help="Max number of words contained in a sentence")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
