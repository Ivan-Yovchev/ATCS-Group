import torch
from torch import nn

class BertMaxBatcher(nn.Module):

    def __init__(self, bert_model, max_len):
        super().__init__()

        self.bert = bert_model
        self.max_len = max_len

    def forward(self, document, mask):

        # Encode sentences in documents

        mask = torch.flatten(mask, start_dim = 0, end_dim = 1)

        # Output (Document x Sentence) x Token x EmbDim
        x = self.bert(
            torch.flatten(
                document,
                start_dim = 0,
                end_dim = 1
            ),
            attention_mask = mask
        )

        # Encode docs (average per sentence)
        x = torch.max(x[0] * mask.unsqueeze(-1), dim=1)[0]

        # Split (Document x Sentence) dim
        x = x.view(*document.shape[:2], *x.shape[1:])

        # Channel in the middle
        x = x.permute(0, 2, 1)
        return torch.nn.functional.pad(x, (0, self.max_len - x.size(2)))

    def train_bert(self):
        return self.bert.train()

    def eval_bert(self):
        return self.bert.eval()

