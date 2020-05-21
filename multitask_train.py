import json
from collections import Counter

from train import *
from datetime import datetime
import pickle as pkl
from torch.utils.data import DataLoader, ConcatDataset
import os
from typing import Mapping, Tuple, List
import logging


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


def mmap_dataset(ds_name, ds_path, args, sent_embedder: BertManager, tokenizer: BertTokenizer, dir='temp'):
    trainfname = os.path.join(dir, ds_name)
    dataset_type = ds_name.split('.')[0]

    if os.path.isfile(trainfname + '.json'):
        logging.info("Found file %s. Loading it.", trainfname)
        dataset = NumpyBackedDataset(trainfname, args.device)
    else:
        logging.info("File %s not found. Creating it.", trainfname)
        dataset = get_dataset(dataset_type, ds_path, tokenizer, args.max_len, args.max_sent, 1, args.device)
        dataset = NumpyBackedDataset(trainfname, args.device, True, *just_apply_bert(dataset, sent_embedder))
    return dataset



def train_multitask(args, ds_names: List[str], ds_dict: Mapping):
    logging.info('Starting train_multitask now.')
    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    # writer = SummaryWriter(f'runs/{args.dataset_type}/{args.batch_size}_{args.max_len}_{args.max_sent}_{args.lr}')
    # dataset types
    d_types = set((name.split('.')[0] for name in ds_names))
    class_and_loss = {d_type: (task_classifier_factory(args, d_type), loss_task_factory(d_type)) for d_type in d_types}
    conv_model = CNNModel(args.embed_size, args.device, n_filters=args.n_filters, batch_norm_eval=True)
    logging.info('Models created.')

    # move to device
    conv_model.to(args.device)
    for tsc, _ in class_and_loss.values():
        tsc.to(args.device)

    # make a list with all parameters to be optimized
    parameters_to_optimize = list(conv_model.parameters())
    for tsc, _ in class_and_loss.values():
        parameters_to_optimize.extend(tsc.parameters())

    optim = torch.optim.Adam(params=parameters_to_optimize, lr=args.lr)
    logging.info('Optimizer created.')

    train_dataloaders = {
        'gcdc': DataLoader(dataset=ConcatDataset([v[0] for k, v in ds_dict.items() if k.split('.')[0] == 'gcdc']),
                           shuffle=True,
                           # num_workers=args.n_workers,
                           batch_size=args.batch_size,
                           collate_fn=NumpyBackedDataset.collate_fn),
        'hyperpartisan': DataLoader(dataset=ds_dict['hyperpartisan'][0],
                                    shuffle=True,
                                    # num_workers=args.n_workers,
                                    batch_size=args.batch_size,
                                    collate_fn=NumpyBackedDataset.collate_fn),
        'fake_news': DataLoader(
            dataset=ConcatDataset([v[0] for k, v in ds_dict.items() if k.split('.')[0] == 'fake_news']),
            shuffle=True,
            # num_workers=args.n_workers,
            batch_size=args.batch_size,
            collate_fn=NumpyBackedDataset.collate_fn),
        'persuasiveness': DataLoader(dataset=ds_dict['persuasiveness'][0],
                                     shuffle=True,
                                     # num_workers=args.n_workers,
                                     batch_size=args.batch_size,
                                     collate_fn=NumpyBackedDataset.collate_fn)
    }
    logging.info('Dataloaders created.')

    # testset dictionary
    testsets_dict = {k: v[1] for k, v in ds_dict.items()}
    logging.info('Testset dictionary created.')
    # initialize the iterators
    train_dl_iters = {k: iter(v) for k, v in train_dataloaders.items()}
    epoch_counter = Counter()

    # test them before training


    # imagine a task sampling here
    for _ in tqdm(range(50)):
        dataset_name = random.choice(ds_names)
        # dataset_name = 'hyperpartisan'
        dataset_type = dataset_name.split('.')[0]
        dl_iter = train_dl_iters[dataset_type]

        logging.info('Sampled dataset: %s', dataset_name)

        try:
            batch = next(dl_iter)
        except StopIteration:
            epoch_counter[dataset_type] += 1
            dl_iter = iter(train_dataloaders[dataset_type])
            train_dl_iters[dataset_type] = dl_iter
            batch = next(dl_iter)

        task_classifier, (binary_classification, loss) = class_and_loss[dataset_type]
        logging.info("Class and loss retrieved.")
        train_one_batch(batch, model=conv_model, task_classifier=task_classifier, loss=loss, optim=optim,
                        device=args.device)
    from pprint import pprint
    from tabulate import tabulate
    ev = eval_model_on_all(conv_model, class_and_loss, testsets_dict, batch_size=args.batch_size)
    print(tabulate(ev, headers='keys'))


def eval_model_on_all(model: nn.Module, classifiers_and_losses: Mapping[str, Tuple[nn.Module, nn.Module]],
                      testsets: Mapping[str, Dataset], batch_size: int) -> Mapping[str, Tuple[float, float]]:
    """

    :param model: the model
    :param classifiers_and_losses: contains the classifiers and losses.
     The variable name would also be a cool indie band name.
    :param testsets:
    :return:
    """
    d = {}
    for t, dt in testsets.items():
        # if t.startswith('fake_news.gossipcop'):
            # continue
        tsc, (binary_class, loss) = classifiers_and_losses[t.split('.')[0]]
        d[t] = eval_model(model, tsc, DataLoader(dt, batch_size=batch_size, collate_fn=NumpyBackedDataset.collate_fn),
                          loss=loss,
                          binary=binary_class)
        logging.info('Results on %s: %.4f %.4f', t, *d[t])
    return d


def train_one_batch(batch, model: nn.Module, task_classifier: nn.Module, loss: nn.Module, optim: torch.optim.Optimizer,
                    device: torch.device):
    optim.zero_grad()
    model.train()
    task_classifier.train()

    x, label = batch
    x, label = x.to(device), label.to(device)
    logging.info('x shape %s', str(x.shape))
    logging.info('label shape %s', str(label.shape))

    out = task_classifier(model(x))
    l = loss(out, label)
    # Backpropagate and update weights
    l.backward()
    optim.step()
    optim.zero_grad()


def main(args):
    time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(f'runs/multitask/{args.batch_size}_{args.max_len}_{args.max_sent}_{args.lr}')

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    sent_embedder = BertManager(bert_model, args.device)

    datasets_dict = {}
    for ds_name, ds_paths in args.dataset_paths.items():
        trainset = mmap_dataset(f'{ds_name}-train', ds_paths['train'], args, sent_embedder, bert_tokenizer,
                                dir=args.temp_dir)
        testset = mmap_dataset(f'{ds_name}-test', ds_paths['test'], args, sent_embedder, bert_tokenizer,
                               dir=args.temp_dir)
        datasets_dict[ds_name] = (trainset, testset)

    train_multitask(args, list(args.dataset_paths.keys()), datasets_dict)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_len", type=int, default=100, help="Max number of words contained in a sentence")
    parser.add_argument("--max_sent", type=int, default=50, help="Max number of sentences per document")
    parser.add_argument("--doc_emb_type", type=str, default="max_batcher", help="Type of document encoder")
    parser.add_argument("--n_filters", type=int, default=128, help="Number of filters for CNN model")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--n_workers", type=int, default=4, help="number of cpu workers for each dataloader")
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
