import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import BertTokenizer

import numpy as np
import tqdm
import wandb
import random
import argparse
import os

import _init_paths

from models.bert_classifier import Bert_Simple_Classifier
from models.query_decoder import Bert_with_Decoder
from utils.dataset_loader import AG_NEWS_TOKENIZED, EURLEX57K
from utils.other import save_checkpoint
from utils.metrics import single_label_count_accurate, PrecisionCounter
from utils.custom_loss import ASLSingleLabel, AsymmetricLossOptimized

# Enable deterministic behaviour
SEED = 101
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global hyper parameters
CACHE_DIR = './data'
CHECKPOINT_PATH = './model_param/state_' + str(random.randint(1000000, 9999999)) + '.pt'
PREC_K = [1, 3, 5]
SINGLE_LABEL_DS = ['AG_NEWS']
MULTI_LABEL_DS = ['EURLEX57K']
NUM_LABELS_DICT = {'AG_NEWS': 4, 'EURLEX57K': 4270}

random.seed(SEED)  # Comes after CHECKPOINT_PATH


def parse_args():
    """
    Parse arguments given to the script.
    Returns:
        The parsed argument object.
    """

    # https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/utils.py#L4

    # TODO: implmenet choices in add_argument in future

    parser = argparse.ArgumentParser(
        description="Single label AG NEWS Training (single GPU ONLY)")
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs (single node)."
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="wandb entity"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AG_NEWS",
        help="EURLEX57K, AG_NEWS"
    )
    parser.add_argument(
        "--model",
        default='bert_classifier',
        type=str,
        choices=['bert_classifier', 'querydecoder'],
        help="model to be used"
    )
    parser.add_argument(
        "--qd_dim",
        default=2048,
        type=int,
        help="querydecoder param: model dimension"
    )
    parser.add_argument(
        "--qd_num_decoder",
        default=2,
        type=int,
        help="querydecoder param: num of decoders"
    )
    parser.add_argument(
        "--qd_num_heads",
        default=4,
        type=int,
        help="querydecoder param: num of heads"
    )
    parser.add_argument(
        "--qd_ffn_dim",
        default=8192,
        type=int,
        help="querydecoder param: dimension of feedforward network in decoder"
    )
    parser.add_argument(
        "--batch",
        default=16,
        type=int,
        help="number of data samples in one batch"
    )
    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        help="number of total epochs to run"
    )
    parser.add_argument(
        "--lr",
        default=0.000001,
        type=lambda x: float(x),
        help="maximal learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0001,
        type=lambda x: float(x),
        help="weight decay l2 regularization"
    )
    parser.add_argument(
        "--bert_type",
        default='bert-base-uncased',
        type=str,
        help="bert type for input into huggingfaceco transformers"
    )
    parser.add_argument(
        "--frozen_bert",
        action='store_true',
        help="set bert parameters learnable"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug mode by setting little datasets"
    )
    parser.add_argument(
        "--loss",
        default='CrossEntropyLoss',
        type=str,
        choices=['CrossEntropyLoss', 'ASLSingleLabel', 'AsymmetricLossOptimized', 'BCEWithLogitsLoss'],
        help="loss fn"
    )
    parser.add_argument(
        "--gamma_neg",
        default=1.0,
        type=lambda x: float(x),
        help="negative gamma param for ASL loss"
    )
    parser.add_argument(
        "--gamma_pos",
        default=0.0,
        type=lambda x: float(x),
        help="positive gamma param for ASL loss"
    )
    parser.add_argument(
        "--learning_rate_style",
        default='constant',
        type=str,
        choices=['constant', 'OneCycleLR'],
        help="learning rate scheduler (constant is default)"
    )
    parser.add_argument(
        "--optimiser",
        default='Adam',
        type=str,
        help="Adam"
    )

    args = parser.parse_args()
    return args


def setup(args):
    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

    os.environ['MASTER_ADDR'] = 'localhost'  # Doesn't matter for single node
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.rank)


def cleanup():
    dist.destroy_process_group()


def start_wandb(args):
    """
    Initialise Weights and Biases
    """

    wandb.login()

    config = dict(
        epochs=args.epochs,
        batch_size=args.batch,
        dataset=args.dataset,
        model=args.model,
        optimiser=args.optimiser,
        optimiser_hyperparams='default',
        frozen_bert=args.frozen_bert,
        learning_rate_style=args.learning_rate_style,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        checkpoint=CHECKPOINT_PATH
    )

    run = wandb.init(project='multilabel-query', entity=args.entity, config=config)

    return run


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler):  # need to add args to here later

    """
    Training loop within a single epoch

    Returns: tuple
        epoch_train_loss_avg: float
        epoch_train_metric: # TODO: fix
    """

    model.train()

    epoch_train_loss_avg = 0
    if args.multilabel:
        epoch_train_metric = PrecisionCounter(k=PREC_K)
    else:
        epoch_train_metric = 0

    for inputs, label in tqdm.tqdm(train_loader):
        # Move all tensors to GPUs
        if args.gpus > 1:
            label = label.cuda(args.rank)
            attn_mask = inputs['attention_mask'].cuda(args.rank)
            input_id = inputs['input_ids'].cuda(args.rank)
        else:
            label = label.cuda()
            attn_mask = inputs['attention_mask'].cuda()
            input_id = inputs['input_ids'].cuda()

        # Typical training loop
        out = model(input_id, attn_mask)
        loss = criterion(out, label)

        # Metrics
        with torch.no_grad():
            epoch_train_loss_avg += loss.item()
            if args.multilabel:
                epoch_train_metric.update(out, label)
            else:
                epoch_train_metric += single_label_count_accurate(out, label)

        # Gradient optimisation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    with torch.no_grad():
        epoch_train_loss_avg /= len(train_loader)
        if args.multilabel:
            epoch_train_metric.finalise(len(train_loader), args.batch)
        else:
            epoch_train_metric /= len(train_loader) * args.batch

    return epoch_train_loss_avg, epoch_train_metric


def validate(args, val_loader, model, criterion, epoch):
    """
    Validation loop within a single epoch

    Returns: tuple
        epoch_val_loss_avg: float
        epoch_val_metric: # TODO: fix
    """

    model.eval()

    with torch.no_grad():

        epoch_val_loss_avg = 0

        if args.multilabel:
            epoch_val_metric = PrecisionCounter(k=PREC_K)
        else:
            epoch_val_metric = 0

        for inputs, label in val_loader:
            # Move all tensors to GPU
            if args.gpus > 1:
                label = label.cuda(args.rank)
                attn_mask = inputs['attention_mask'].cuda(args.rank)
                input_id = inputs['input_ids'].cuda(args.rank)
            else:
                label = label.cuda()
                attn_mask = inputs['attention_mask'].cuda()
                input_id = inputs['input_ids'].cuda()

            # Run model and calculate loss on val
            out = model(input_id, attn_mask)
            loss = criterion(out, label)

            # Metrics
            epoch_val_loss_avg += loss.item()

            if args.multilabel:
                epoch_val_metric.update(out, label)
            else:
                epoch_val_metric += single_label_count_accurate(out, label)

        epoch_val_loss_avg /= len(val_loader)
        if args.multilabel:
            epoch_val_metric.finalise(len(val_loader), args.batch)
        else:
            epoch_val_metric /= len(val_loader) * args.batch

    return epoch_val_loss_avg, epoch_val_metric


def train_loop(args, model, device, train_loader, val_loader, sampler, run):
    """
    Training and validation loop
    """

    # Loss function
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ASLSingleLabel':
        criterion = ASLSingleLabel(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
    elif args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'AsymmetricLossOptimized':
        criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
    else:
        raise NotImplementedError

    criterion.cuda(args.rank)

    # Initial learning rate
    if args.learning_rate_style == 'constant':
        initial_lr = args.lr
    elif args.learning_rate_style == 'OneCycleLR':
        initial_lr = args.lr * args.batch / 256
    else:
        raise NotImplementedError

    # Optimiser
    if args.optimiser == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               initial_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # Scheduler
    if args.learning_rate_style == 'constant':
        scheduler = None
    elif args.learning_rate_style == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                  steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.3)
    else:
        raise NotImplementedError

    # Use validation loss to checkpoint model state (assume initialises larger than 0, i.e. larger better)
    best_metric = 0

    for epoch in range(args.epochs):

        if args.gpus > 1:
            sampler.set_epoch(epoch)

        train_loss, train_metric = train(args, train_loader, model, criterion, optimizer, epoch, scheduler)

        val_loss, val_metric = validate(args, val_loader, model, criterion, epoch)

        if args.rank == 0 or args.rank is None:

            if args.multilabel:

                train_prec_temp = {'train_prec_at_' + str(key): value for (key, value) in train_metric.dict().items()}
                val_prec_temp = {'val_prec_at_' + str(key): value for (key, value) in val_metric.dict().items()}

                run.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                           **train_prec_temp, **val_prec_temp})

                val_metric_single = val_metric.dict()[1]
            else:
                run.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                       'train_acc': train_metric, 'val_acc': val_metric})
                val_metric_single = val_metric

            # Checkpoint
            if val_metric_single > best_metric:
                best_metric = val_metric_single

                run.summary["best_val_metric"] = best_metric

                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'best_metric': best_metric}, filename=CHECKPOINT_PATH)

                print("Best metric achieved.")

            print(
                f'Epoch: {epoch + 1} | Train Loss: {train_loss: .3f} | Val Loss: {val_loss: .3f}'
            )

            if args.multilabel:
                print('Training P@K: ', train_metric)
                print('Validation P@K: ', val_metric)
            else:
                print(
                    f'Train Acc: {train_metric: .3f} | Val Acc: {val_metric: .3f}'
                )


def main(rank, args):

    args.rank = rank

    if args.rank == 0 or args.rank is None:
        RUN = start_wandb(args)
    else:
        RUN = None

    if args.gpus > 1:
        setup(args)
        args.batch = int(args.batch / args.gpus)

    if args.dataset in SINGLE_LABEL_DS:
        args.multilabel = False
    elif args.dataset in MULTI_LABEL_DS:
        args.multilabel = True
    else:
        raise NotImplementedError

    # Tokenize and load dataset
    tokenizer = BertTokenizer.from_pretrained(args.bert_type)

    # Debug mode
    if args.debug:
        dataset_size_train = 600
        dataset_size_val = 100
    else:
        dataset_size_train = None
        dataset_size_val = None

    if args.dataset == 'AG_NEWS':
        train_iter = AG_NEWS_TOKENIZED('./data/AG_NEWS_MOD/', train=True, tokenizer=tokenizer, N=dataset_size_train)
        val_iter = AG_NEWS_TOKENIZED('./data/AG_NEWS_MOD/', train=False, tokenizer=tokenizer, N=dataset_size_val)
    elif args.dataset =='EURLEX57K':
        train_iter = EURLEX57K(type='train', tokenizer=tokenizer, N=dataset_size_train)
        val_iter = EURLEX57K(type='dev', tokenizer=tokenizer, N=dataset_size_val)
    else:
        raise NotImplementedError

    if args.gpus > 1:
        TRAIN_SAMPLER = torch.utils.data.distributed.DistributedSampler(train_iter, num_replicas=args.gpus,
                                                                        rank=args.rank, shuffle=False,
                                                                        drop_last=True)
    else:
        TRAIN_SAMPLER = None

    TRAIN_LOADER = DataLoader(train_iter, batch_size=args.batch, drop_last=True, shuffle=False,
                              sampler=TRAIN_SAMPLER)
    VAL_LOADER = DataLoader(val_iter, batch_size=args.batch, drop_last=True, shuffle=False)

    # Model init
    if args.model == 'bert_classifier':
        MODEL = Bert_Simple_Classifier(num_labels=NUM_LABELS_DICT[args.dataset],
                                       frozen_bert=args.frozen_bert, bert_type=args.bert_type)
    elif args.model == 'querydecoder':
        MODEL = Bert_with_Decoder(num_labels=NUM_LABELS_DICT[args.dataset], d_model=args.qd_dim, num_decoder=args.qd_num_decoder,
                                  n_heads=args.qd_num_heads,
                                  frozen_bert=args.frozen_bert, bert_type=args.bert_type,
                                  dropout=0.1, decoder_ffn_dim=args.qd_ffn_dim)
    else:
        raise NotImplementedError

    if args.gpus > 1:
        MODEL.cuda(args.rank)
        MODEL = DDP(MODEL, device_ids=[args.rank], output_device=args.rank)
    else:
        MODEL.cuda()

    if args.rank == 0 or args.rank is None:
        RUN.watch(MODEL)

    train_loop(args, MODEL, DEVICE, TRAIN_LOADER, VAL_LOADER, TRAIN_SAMPLER, RUN)


if __name__ == "__main__":
    args = parse_args()

    args.world_size = args.gpus * 1  # As we are only using a single node

    assert torch.cuda.device_count() >= args.world_size, "GPU required less than available"

    if args.gpus > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size)
    else:
        main(None, args)
