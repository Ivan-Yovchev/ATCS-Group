import json
from train import *
from datetime import datetime
import pickle as pkl
import os

def load_datasets(ds_name, ds_paths, args, sent_embedder: BertManager, tokenizer: BertTokenizer):
    filename = f'temp/{ds_name}.pt'
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)
        
    dataset_type = ds_name.split('.')[0]
    dataset = get_dataset(dataset_type, ds_paths['train'], tokenizer, args.max_len, args.max_sent,
                        args.batch_size if args.finetune else 1, args.device)
    testset = get_dataset(dataset_type, ds_paths['test'], tokenizer, args.max_len, args.max_sent,
                        args.batch_size if args.finetune else 1, args.device)
    dataset = BertPreprocessor(dataset, sent_embedder, batch_size=args.batch_size)
    testset = BertPreprocessor(testset, sent_embedder, batch_size=args.batch_size)
    
    with open(filename, 'wb') as f:
        torch.save([dataset, testset], f)
    
    return dataset, testset

def train(args):
    valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss, binary=binary_classification)
    print(f'Initial acc: {valid_acc:.4f} loss: {valid_loss:.4f}')
    best_acc = 0
    # optim = transformers.optimization.AdamW(list(model.parameters()) + list(bert_model.parameters()), args.lr)
    optim = torch.optim.Adam(list(model.parameters()) + list(task_classifier.parameters()), args.lr)
    # optim = transformers.optimization.AdamW(list(conv_model.parameters()), args.lr)

    lr_scheduler = ReduceLROnPlateau(optim, mode='max', patience=5, factor=0.8)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.8)

    for epoch in range(args.n_epochs):

        if optim.defaults['lr'] < 1e-6: break
        train_acc, train_loss = train_model(model, task_classifier, dataset, loss, optim, binary=binary_classification)
        valid_acc, valid_loss = eval_model(model, task_classifier, testset, loss, binary=binary_classification)
        print(f'Epoch {epoch:02d}: train acc: {train_acc:.4f}'
              f' train loss: {train_loss:.4f} valid acc: {valid_acc:.4f}'
              f' valid loss: {valid_loss:.4f}')

        lr_scheduler.step(valid_acc)

        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_acc', valid_acc, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)

        if best_acc < valid_acc:
            best_acc = valid_acc

            with open(os.path.join('models', f"{args.dataset_type}.{time_log}.pt"), 'wb') as f:
                torch.save({
                    'cnn_model': conv_model.state_dict(),
                    'bert_model': bert_model.state_dict(),
                    'task_classifier': task_classifier.state_dict(),
                    'epoch': epoch
                }, f)

def main(args):
    for f in os.listdir(args.temp_dir):
        os.remove(os.path.join(args.temp_dir, f))
    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(f'runs/multitask/{args.batch_size}_{args.max_len}_{args.max_sent}_{args.lr}')
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    sent_embedder = BertManager(bert_model, args.max_len, args.device)

    for ds_name, ds_paths in args.dataset_paths.items():
        trainset, testset = load_datasets(ds_name, ds_paths, args, sent_embedder, bert_tokenizer)
        print('ts', len(trainset), trainset.get_n_classes(), len(testset), testset.get_n_classes())
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=100, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=50, help="Max number of sentences per document")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda', help="device to use for the training")
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False,
                        help="Set to true to fine tune bert")
    args = parser.parse_args()
    args.temp_dir = 'temp'
    args.embed_size = 768
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.temp_dir, exist_ok=True)
    with open('multi_task.json', 'r') as f:
        args.dataset_paths = json.load(f)
    # print('args', args)
    main(args)

