import os
import json

from data_utils import read_data, process_len

train_data = read_data('public_dat/train_refind_official.json')
process_len(train_data)
dev_data = read_data('public_dat/dev_refind_official.json')
process_len(dev_data)
test_data = read_data('public_dat/test_refind_official.json')
process_len(test_data)
print(len(train_data), len(dev_data), len(test_data))

with open('public_dat_truncate/train_refind_official.json', 'w') as ouf:
    json.dump(train_data, ouf, indent=4)

with open('public_dat_truncate/dev_refind_official.json', 'w') as ouf:
    json.dump(dev_data, ouf, indent=4)

with open('public_dat_truncate/test_refind_official.json', 'w') as ouf:
    json.dump(test_data, ouf, indent=4)
