import transformers
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import logging
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from collections import defaultdict
from torch import nn, optim
import json
import random
import os
import argparse

from transformers import AutoModel, AutoTokenizer 
from data_utils import read_data, REDataset, create_data_loader, train_ 
from model import REModel 

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./public_dat_truncate', type=str, help="dir of data")
parser.add_argument('--model_name', default='bert-base-uncased', type=str, help="bert-base-uncased or vinai/bertweet-base vinai/bertweet-large")
parser.add_argument('--af', default='max', type=str, help="aggregating function: head, avg or max")
parser.add_argument('--max_len', default=256, type=int, help="max len for the input text")
parser.add_argument('--num_epochs', default=50, type=int, help="Training epochs")
parser.add_argument('--batch_size', default=8, type=int, help="batch size")
parser.add_argument('--inter', default=120, type=int, help="inter for printing training logs")
parser.add_argument('--early_stop', default=15, type=int, help="stop if the eval performance does not rise")
parser.add_argument('--warmup_rate', default=0.06, type=float)
parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
parser.add_argument('--save_dir', default='saved_models', type=str, help="save model at save_path")
args = parser.parse_args()

def get_logger(output_dir=None):
    # remove previous handlers
    root = logging.root
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler("{}/log.txt".format(output_dir), mode="a", delay=False),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('FinRE')
    logger.setLevel(10)
    return logger

logger = get_logger(args.save_dir)

# Load and massage the dataframes.
train_path = args.data_dir+'/train_refind_official.json'
dev_path = args.data_dir+'/dev_refind_official.json'
test_path = args.data_dir+'/test_refind_official.json'

train_data = read_data(train_path)
dev_data = read_data(dev_path)
test_data = read_data(test_path)

logger.info("train: %s dev: %s, test: %s" % (len(train_data), len(dev_data), len(test_data)))
# dict_keys(['id', 'docid', 'relation', 'rel_group', 'token', 'e1_start', 'e1_end', 'e2_start', 'e2_end', 'e1_type', 'e2_type', 'spacy_pos', 'spacy_ner', 'spacy_head', 'spacy_deprel', 'sdp', 'sdp_tok_idx', 'e1', 'e2', 'token_test', 'truncate'])

# Instantiate the tokenizer.

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_data_loader = create_data_loader(
    train_data, tokenizer, args.max_len, args.batch_size
)
val_data_loader = create_data_loader(
    dev_data, tokenizer, args.max_len, args.batch_size
)
test_data_loader = create_data_loader(
    test_data, tokenizer, args.max_len, args.batch_size
)

model = REModel(args).cuda()

# Configure the optimizer and scheduler.
optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
total_steps = len(train_data_loader) * args.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_rate*total_steps, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss().cuda()

best_eval_acc, best_eval_f1, best_test_acc, best_test_f1 = train_(args, model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, scheduler, logger)
results_dict = {"best_eval_acc": best_eval_acc, "best_eval_f1": best_eval_f1, "best_test_acc": best_test_acc, "best_test_f1": best_test_f1}

with open(args.save_dir+'/results.json', 'w') as f:
    json.dump(results_dict, f)
