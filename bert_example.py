import torch

from torch.utils.data import DataLoader
from torch import nn

from transformers import BertModel, BertTokenizer
from datasets import GCDC_Dataset, collate_pad_fn
from cnn_model import CNNModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, Davide is gay.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = bert_model(input_ids)

data_path = 'data/GCDC/Clinton_test.csv'
max_len = 300

dataset = GCDC_Dataset(data_path, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_pad_fn)

batch = next(iter(dataloader))
batch_output = bert_model(batch[0])

print(outputs[0].shape)
print(batch_output[0].permute(0,2,1).shape)

### Model test

n_filters = 128
classifier = nn.Sequential(nn.Linear(5*n_filters, 1), nn.Sigmoid())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
model = CNNModel(batch_output[0].shape[2], max_len, classifier, device, n_filters=n_filters)
print(model(batch_output[0].permute(0,2,1).to(device)))
