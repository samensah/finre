import os
import json


def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


test_data = read_data('public_dat_truncate/test_refind_official.json')
train_data = read_data('public_dat_truncate/train_refind_official.json')
dev_data = read_data('public_dat_truncate/dev_refind_official.json')

# dict_keys(['id', 'docid', 'relation', 'rel_group', 'token', 'e1_start', 'e1_end', 'e2_start', 'e2_end', 'e1_type', 'e2_type', 'spacy_pos', 'spacy_ner', 'spacy_head', 'spacy_deprel', 'sdp', 'sdp_tok_idx', 'e1', 'e2', 'token_test'])
# {'id': 'BERTPretrain_10KReports/2017/QTR1/20170330_10-K_edgar_data_1667840_0001571049-17-003036_1.txt', 'docid': '2016/2017', 'relation': 'no_relation', 'rel_group': 'ORG-ORG', 'token': ['other', 'changes', 'in', 'the', 'financial', 'condition', 'or', 'future', 'prospects', 'of', 'issuers', 'of', 'securities', 'that', 'Best', 'Hometown', 'Bancorp', ',', 'Inc.', 'own', ',', 'including', 'Best', 'Hometown', 'Bancorp', ',', 'Inc.', 'stock', 'in', 'the', 'Federal', 'Home', 'Loan', 'Bank', '(', 'FHLB', ')', 'of', 'Chicago', 'or', 'FHLB', 'and', '.'], 'e1_start': 22, 'e1_end': 27, 'e2_start': 40, 'e2_end': 41, 'e1_type': 'ORG', 'e2_type': 'ORG', 'spacy_pos': ['JJ', 'NNS', 'IN', 'DT', 'JJ', 'NN', 'CC', 'JJ', 'NNS', 'IN', 'NNS', 'IN', 'NNS', 'WDT', 'NNP', 'NNP', 'NNP', ',', 'NNP', 'VBP', ',', 'VBG', 'NNP', 'NNP', 'NNP', ',', 'NNP', 'NN', 'IN', 'DT', 'NNP', 'NNP', 'NNP', 'NNP', '-LRB-', 'NNP', '-RRB-', 'IN', 'NNP', 'CC', 'NNP', 'CC', '.'], 'spacy_ner': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O'], 'spacy_head': [1, 1, 1, 5, 5, 2, 5, 8, 5, 5, 9, 10, 11, 19, 18, 18, 18, 18, 19, 12, 12, 12, 26, 26, 26, 26, 27, 21, 27, 33, 33, 33, 33, 28, 35, 33, 35, 33, 37, 33, 33, 27, 1], 'spacy_deprel': ['amod', 'ROOT', 'prep', 'det', 'amod', 'pobj', 'cc', 'amod', 'conj', 'prep', 'pobj', 'prep', 'pobj', 'dobj', 'compound', 'compound', 'nmod', 'punct', 'nsubj', 'relcl', 'punct', 'prep', 'nmod', 'nmod', 'nmod', 'punct', 'compound', 'pobj', 'prep', 'det', 'compound', 'compound', 'compound', 'pobj', 'punct', 'appos', 'punct', 'prep', 'pobj', 'cc', 'conj', 'cc', 'punct'], 'sdp': ['Best Hometown Bancorp , Inc.', 'stock', 'Bank', 'Chicago', 'FHLB'], 'sdp_tok_idx': [27, 33, 38], 'e1': 'besthometownbancorpinc', 'e2': 'fhlb', 'token_test': 'otherchangesinthefinancialconditionorfutureprospectsofissuersofsecuritiesthatbesthometownbancorpincownincludingbesthometownbancorpincstockinthefederalhomeloanbankfhlbofchicagoorfhlband'}

lens_list = []
for d in train_data+test_data+dev_data:
    lens_list.append(len(d['token']))


def check_count(lens_list, low, high):
    count = 0
    for l in lens_list:
        if l > low and l <= high:
            count += 1
    print(count, len(lens_list))
    print(count/len(lens_list))


check_count(lens_list, 0, 64)
check_count(lens_list, 0, 130)
