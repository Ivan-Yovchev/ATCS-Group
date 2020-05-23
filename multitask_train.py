import argparse
import torch
from transformers import BertTokenizer, BertModel
from metatrain import get_dataset_paths


def main():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    fake_news_desc, gcdc_desc, partisan_desc, pers_desc = get_dataset_paths(args.dataset_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_len", type=int, default=15, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=15, help="Max number of sentences in a doc")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    # parser.add_argument("--embed_size", type=int, default=768, help="Embedding size") # It is the output of BERT,
    # cannot be changed!
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training model in specific episode")
    parser.add_argument("--meta_lr", type=float, default=0.0001, help="Learning rate for updateing meta model")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--dataset_json", type=str, default='./dataset-paths.json',
                        help="JSON file containing the dataset paths")

    parser.add_argument("--meta_epochs", type=int, default=5, help="Number of meta epochs")
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False,
                        help="Set to true to fine tune bert")
    parser.add_argument("--train_size_support", type=int, default=8, help="Size of support set during training")
    parser.add_argument("--train_size_query", type=int, default=8, help="Size of query set during training")
    # parser.add_argument("--shots", type=int, default=8, help="Number of examples during meta validation/testing")
    parser.add_argument("--kernels", type=lambda x: [int(i) for i in x.split(',')], default="2,4,6",
                        help="Kernel sizes per cnn block")

    args = parser.parse_args()
    args.embed_size = 768
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    main(args)

