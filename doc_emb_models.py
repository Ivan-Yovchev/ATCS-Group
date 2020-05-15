import torch
from torch import nn


class BertManager(nn.Module):

    def __init__(self, bert_model, max_len, device, mode="cls"):
        super().__init__()

        self.bert = bert_model
        self.max_len = max_len
        self.mode = mode
        self.device = device

        self.to(device)

    def forward(self, document, mask):
        document = document.to(self.device)
        mask = mask.to(self.device)
        # Encode sentences in documents

        mask = torch.flatten(mask, start_dim=0, end_dim=1)

        # Output (Document x Sentence) x Token x EmbDim
        x = self.bert(
            torch.flatten(
                document,
                start_dim=0,
                end_dim=1
            ),
            attention_mask=mask
        )

        # Encode docs (average per sentence)
        if self.mode == "cls":
            x = x[1]
        elif self.mode == "max":
            x = torch.max(x[0] * mask.unsqueeze(-1), dim=1)[0]
        elif self.mode == "mean":
            x = torch.sum(x[0] * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1).unsqueeze(-1)

        # Split (Document x Sentence) dim
        x = x.view(*document.shape[:2], *x.shape[1:])

        # Channel in the middle
        x = x.permute(0, 2, 1)
        return torch.nn.functional.pad(x, (0, self.max_len - x.size(2)))

    def train(self):
        return self.bert.train()

    def eval(self):
        return self.bert.eval()


if __name__ == "__main__":
    from transformers import BertModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertManager(bert_model, 200, device)

    x = torch.LongTensor(2, 7, 20).random_(0, 3000)
    mask = torch.LongTensor(2, 7, 20).random_(0, 2)

    print(model(x, mask).shape)
