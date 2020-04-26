from transformers import BertModel, BertTokenizer
from datasets import GCDC_Dataset, collate_pad_fn
import torch
from torch.utils.data import DataLoader

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, Davide is gay.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

data_path = 'data/GCDC/Clinton_train.csv'
dataset = GCDC_Dataset(data_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_pad_fn)

batch = next(iter(dataloader))
batch_output = model(batch[0])

print(outputs[0].shape)
print(batch_output[0].shape)