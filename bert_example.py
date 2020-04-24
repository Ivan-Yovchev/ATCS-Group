from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, Davide is gay.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

print(outputs[0].shape)