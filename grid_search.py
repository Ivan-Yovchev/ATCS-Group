import argparse
import torch
import traceback
from train import main as train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="data/GCDC/Clinton_train.csv", help="Path to training data")
    parser.add_argument("--test_path", type=str, default="data/GCDC/Clinton_test.csv", help="Path to testing data")
    parser.add_argument("--dataset_type", type=str, default="gcdc", help="Dataset type")
    parser.add_argument("--kfold", type=bool, default=False,
                        help="10fold for hyperpartisan dataset. test_path value will be ignored")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False,
                        help="Set to true to fine tune bert")
    args = parser.parse_args()
    args.embed_size = 768
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for lr in [1e-1, 1e-3, 1e-4, 1e-5]:
        for batch_size in [2, 8, 64]:
            for max_len in [50, 100, 200]:
                for max_sent in [50, 100, 200]:
                    args.lr = lr
                    args.batch_size = batch_size
                    args.max_len = max_len
                    args.max_sent = max_sent
                    try:
                        train_model(args)
                    except Exception as e:
                        print(f'There was an error with the following parameters lr: {lr}, bs: {batch_size}, max_len: {max_len}, max_sent: {max_sent}, exception:{traceback.format_exc()}')
                        continue
                    