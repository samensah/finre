import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score
import json
from constant import label2id, id2label, E1, E2, E1_, E2_

def insert_special_token(d):
    token = d['token']
    if d['e1_start'] < d['e2_start']:
        token = token[:d['e1_start']]+['[e1]']+token[d['e1_start']:d['e1_end']]+['[e1/]']+token[d['e1_end']:]
        d['e1_start'] = d['e1_start']+1
        d['e1_end'] = d['e1_end']+1
        d['e2_start'] = d['e2_start']+2
        d['e2_end'] = d['e2_end']+2
        token = token[:d['e2_start']]+['[e2]']+token[d['e2_start']:d['e2_end']]+['[e2/]']+token[d['e2_end']:]
        d['e2_start'] = d['e2_start']+1
        d['e2_end'] = d['e2_end']+1
        d['token'] = token
    elif d['e1_start'] > d['e2_start']:
        token = token[:d['e2_start']]+['[e2]']+token[d['e2_start']:d['e2_end']]+['[e2/]']+token[d['e2_end']:]
        d['e2_start'] = d['e2_start']+1
        d['e2_end'] = d['e2_end']+1
        d['e1_start'] = d['e1_start']+2
        d['e1_end'] = d['e1_end']+2
        token = token[:d['e1_start']]+['[e1]']+token[d['e1_start']:d['e1_end']]+['[e1/]']+token[d['e1_end']:]
        d['e1_start'] = d['e1_start']+1
        d['e1_end'] = d['e1_end']+1
        d['token'] = token
    else:
        assert True == False

def find_sub_list(ori_list, target_list):
    for i in range(len(ori_list)):
        if ori_list[i:i+len(target_list)] == target_list:
            return i
    return -1

# Construct the dataset.
class REDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_len
    ):
        """
        Downstream code expects reviews and targets to be NumPy arrays.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        for d in self.data:
            insert_special_token(d)

        # tokenize
        for d in self.data:
            d['wp_token'] = self.tokenizer.tokenize(' '.join(d['token']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # dict_keys(['id', 'docid', 'relation', 'rel_group', 'token', 'e1_start', 'e1_end', 'e2_start', 'e2_end', 'e1_type', 'e2_type', 'spacy_pos', 'spacy_ner', 'spacy_head', 'spacy_deprel', 'sdp', 'sdp_tok_idx', 'e1', 'e2', 'token_test', 'truncate'])

        # insert special tokens for entities: [e1] [e1/] [e2] [e2/]
        # ['[', 'e', '##1', ']'], ['[', 'e', '##1', '/', ']'], ['[', 'e', '##2', ']'], ['[', 'e', '##2', '/', ']']
        d = self.data[item]
        wp_token = ['[CLS]']+d['wp_token']+['[SEP]']

        # label
        label = label2id[d['relation']]

        input_ids = self.tokenizer.convert_tokens_to_ids(wp_token)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        attention_mask = len(input_ids) * [1]
        input_ids = input_ids + (self.max_len-len(input_ids))*[0]
        attention_mask = attention_mask + (self.max_len-len(attention_mask))*[0]

        assert len(input_ids) == len(attention_mask)

        e1_start = find_sub_list(input_ids, E1)
        e1_end = find_sub_list(input_ids, E1_)

        assert e1_start != -1 and e1_end != -1

        e2_start = find_sub_list(input_ids, E2)
        e2_end = find_sub_list(input_ids, E2_)

        e1_mask = torch.tensor([0]*len(input_ids), dtype=torch.long)
        e2_mask = torch.tensor([0]*len(input_ids), dtype=torch.long)
        e1_mask[e1_start:e1_end] = 1
        e2_mask[e2_start:e2_end] = 1

        assert e2_start != -1 and e2_end != -1

        return {
            "text": ' '.join(d['token']),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
            "e1_pos": torch.tensor([e1_start, e1_end], dtype=torch.long),
            "e2_pos": torch.tensor([e2_start, e2_end], dtype=torch.long),
            "e1_mask": e1_mask,
            "e2_mask": e2_mask,
        }


# Construct the data loaders.
def create_data_loader(data, tokenizer, max_len, batch_size):
    ds = REDataset(
        data=data, 
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size)


def train_(args, model, train_dataloader, eval_dataloader, test_dataloader, loss_fn, optimizer, scheduler, logger):
    logger.info("evaluate before training...")
    eval_micro_f1, eval_macro_f1, eval_loss = eval_model(model, eval_dataloader, loss_fn)
    logger.info(f"eval micro f1: {eval_micro_f1}, eval macro f1: {eval_macro_f1}, eval loss: {eval_loss}")

    steps = 0
    best_eval_micro_f1, best_eval_macro_f1, best_test_micro_f1, best_test_macro_f1 = -1, -1, -1, -1
    correct_predictions, n_examples, tolerance = 0, 0, 0

    for _ in range(args.num_epochs):
        model = model.train()
        losses = []
        for d in train_dataloader:
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            targets = d["label"].cuda()
            e1_pos = d["e1_pos"].cuda()
            e2_pos = d["e2_pos"].cuda()
            e1_mask = d["e1_mask"].cuda()
            e2_mask = d["e2_mask"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, e1_pos=e1_pos, e2_pos=e2_pos, e1_mask=e1_mask, e2_mask=e2_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            # filter no_relation for computing the acc
            for p,t in zip(preds.tolist(), targets.tolist()):
                if p == t and p == 0:
                    continue
                else:
                    if p == t:
                        correct_predictions += 1
                    n_examples += 1
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if steps % args.inter == 0:
                logger.info(f"loss: {np.mean(losses)}")
            steps += 1

        logger.info(f"Training ACC: {correct_predictions / n_examples}")
        eval_micro_f1, eval_macro_f1, eval_loss = eval_model(model, eval_dataloader, loss_fn)
        logger.info(f"eval micro f1: {eval_micro_f1}, eval macro f1: {eval_macro_f1}, eval loss: {eval_loss}")
        if eval_macro_f1 > best_eval_macro_f1:
            tolerance = 0
            logger.info("saving best model!")
            torch.save(model.state_dict(), args.save_dir+'/best_model.pt')
            test_micro_f1, test_macro_f1, test_loss = eval_model(model, test_dataloader, loss_fn)
            logger.info(f"test micro f1: {test_micro_f1}, test macro f1: {test_macro_f1}, test loss: {test_loss}")
            best_eval_micro_f1 = eval_micro_f1
            best_eval_macro_f1 = eval_macro_f1
            best_test_micro_f1 = test_micro_f1
            best_test_macro_f1 = test_macro_f1
        else:
            tolerance += 1
            test_micro_f1, test_macro_f1, test_loss = eval_model(model, test_dataloader, loss_fn)
            logger.info(f"test micro f1: {test_micro_f1}, test macro f1: {test_macro_f1}, test loss: {test_loss}")

        if tolerance > args.early_stop:
            break

    return best_eval_micro_f1, best_eval_macro_f1, best_test_micro_f1, best_test_macro_f1

def eval_model(model, data_loader, loss_fn):
    model = model.eval()

    losses, predictions, labels = [], [], []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            targets = d["label"].cuda()
            e1_pos = d["e1_pos"].cuda()
            e2_pos = d["e2_pos"].cuda()
            e1_mask = d["e1_mask"].cuda()
            e2_mask = d["e2_mask"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, e1_pos=e1_pos, e2_pos=e2_pos, e1_mask=e1_mask, e2_mask=e2_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            losses.append(loss.item())
            # filter no_relation
            for p,t in zip(preds.tolist(), targets.tolist()):
                if p == t and p == 0:
                    continue
                else:
                    predictions.append(p)
                    labels.append(t)

    eval_loss = np.mean(losses)
    micro_f1 = f1_score(labels, predictions, average="micro")
    macro_f1 = f1_score(labels, predictions, average="macro")

    return micro_f1, macro_f1, eval_loss

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_len(data, max_len=128):
    process_count = 0
    for i in range(len(data)):
        if len(data[i]['token']) <= max_len:
            data[i]['truncate'] = 0
            continue
        else:
            data[i]['truncate'] = 1
            process_count += 1
            if data[i]['e1_start'] < data[i]['e2_start']:
                str_start_e1 = data[i]['e1_start'] - max_len//4
                str_end_e1 = data[i]['e1_start'] + max_len//4 + 1
                if str_start_e1 < 0:
                    str_start_e1 = 0
                if str_end_e1 > len(data[i]['token']):
                    str_end_e1 = len(data[i]['token'])

                str_start_e2 = data[i]['e2_start'] - max_len//4
                str_end_e2 = data[i]['e2_start'] + max_len//4 + 1
                if str_start_e2 < 0:
                    str_start_e2 = 0
                if str_end_e2 > len(data[i]['token']):
                    str_end_e2 = len(data[i]['token'])

                if str_start_e2 >= str_end_e1:
                    new_token = data[i]['token'][str_start_e1:str_end_e1] + data[i]['token'][str_start_e2:str_end_e2]
                    new_e1_start = data[i]['e1_start'] - str_start_e1
                    new_e1_end = data[i]['e1_end'] - str_start_e1
                    new_e2_start = data[i]['e2_start'] - str_start_e2 + (str_end_e1 - str_start_e1)
                    new_e2_end = data[i]['e2_end'] - str_start_e2 + (str_end_e1 - str_start_e1)

                    new_spacy_pos = data[i]['spacy_pos'][str_start_e1:str_end_e1] + data[i]['spacy_pos'][str_start_e2:str_end_e2]
                    new_spacy_ner = data[i]['spacy_ner'][str_start_e1:str_end_e1] + data[i]['spacy_ner'][str_start_e2:str_end_e2]
                else:
                    new_token = data[i]['token'][str_start_e1:str_end_e2]
                    new_e1_start = data[i]['e1_start'] - str_start_e1
                    new_e1_end = data[i]['e1_end'] - str_start_e1
                    new_e2_start = data[i]['e2_start'] - str_start_e2 + (str_end_e1 - str_start_e1) - (str_end_e1 - str_start_e2)
                    new_e2_end = data[i]['e2_end'] - str_start_e2 + (str_end_e1 - str_start_e1) - (str_end_e1 - str_start_e2)

                    new_spacy_pos = data[i]['spacy_pos'][str_start_e1:str_end_e2]
                    new_spacy_ner = data[i]['spacy_ner'][str_start_e1:str_end_e2]
            else:
                str_start_e2 = data[i]['e2_start'] - max_len//4
                str_end_e2 = data[i]['e2_start'] + max_len//4 + 1
                if str_start_e2 < 0:
                    str_start_e2 = 0
                if str_end_e2 > len(data[i]['token']):
                    str_end_e2 = len(data[i]['token'])

                str_start_e1 = data[i]['e1_start'] - max_len//4
                str_end_e1 = data[i]['e1_start'] + max_len//4 + 1
                if str_start_e1 < 0:
                    str_start_e1 = 0
                if str_end_e1 > len(data[i]['token']):
                    str_end_e1 = len(data[i]['token'])

                if str_start_e1 >= str_end_e2:
                    new_token = data[i]['token'][str_start_e2:str_end_e2] + data[i]['token'][str_start_e1:str_end_e1]
                    new_e2_start = data[i]['e2_start'] - str_start_e2
                    new_e2_end = data[i]['e2_end'] - str_start_e2
                    new_e1_start = data[i]['e1_start'] - str_start_e1 + (str_end_e2 - str_start_e2)
                    new_e1_end = data[i]['e1_end'] - str_start_e1 + (str_end_e2 - str_start_e2)

                    new_spacy_pos = data[i]['spacy_pos'][str_start_e2:str_end_e2] + data[i]['spacy_pos'][str_start_e1:str_end_e1]
                    new_spacy_ner = data[i]['spacy_ner'][str_start_e2:str_end_e2] + data[i]['spacy_ner'][str_start_e1:str_end_e1]
                else:
                    new_token = data[i]['token'][str_start_e2:str_end_e1]
                    new_e2_start = data[i]['e2_start'] - str_start_e2
                    new_e2_end = data[i]['e2_end'] - str_start_e2
                    new_e1_start = data[i]['e1_start'] - str_start_e1 + (str_end_e2 - str_start_e2) - (str_end_e2 - str_start_e1)
                    new_e1_end = data[i]['e1_end'] - str_start_e1 + (str_end_e2 - str_start_e2) - (str_end_e2 - str_start_e1)

                    new_spacy_pos = data[i]['spacy_pos'][str_start_e2:str_end_e1]
                    new_spacy_ner = data[i]['spacy_ner'][str_start_e2:str_end_e1]

            data[i]['token'] = new_token
            data[i]['e1_start'] = new_e1_start
            data[i]['e1_end'] = new_e1_end
            data[i]['e2_start'] = new_e2_start
            data[i]['e2_end'] = new_e2_end
            data[i]['spacy_pos'] = new_spacy_pos
            data[i]['spacy_ner'] = new_spacy_ner

    print("process len count:", process_count)

