import os
import json

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

test_data = read_data('public_dat_truncate/test_refind_official.json')
train_data = read_data('public_dat_truncate/train_refind_official.json')
dev_data = read_data('public_dat_truncate/dev_refind_official.json')

rel_list = []
for d in train_data+test_data+dev_data:
    if d['relation'] not in rel_list:
        rel_list.append(d['relation'])

id2label = {i:r for i,r in enumerate(rel_list)}
label2id = {r:i for i,r in enumerate(rel_list)}
print(id2label)
print(label2id)
