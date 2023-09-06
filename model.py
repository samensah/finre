import transformers
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel
)
import torch
import torch.nn.functional as F
import torch.nn as nn
from constant import label2id

# Construct and instantiate the classifier.
class REModel(nn.Module):
    def __init__(self, args):
        super(REModel, self).__init__()

        self.bert = AutoModel.from_pretrained(args.model_name)
        self.out = nn.Linear(self.bert.config.hidden_size*2, len(label2id))
        self.args = args

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos, e1_mask, e2_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state

        if self.args.af == 'head':
            e1_start = e1_pos[:,:1]
            e2_start = e2_pos[:,:1]
            e1_rep = outputs.gather(1, e1_start.unsqueeze(-1).repeat(1,1,outputs.size(2)))
            e2_rep = outputs.gather(1, e2_start.unsqueeze(-1).repeat(1,1,outputs.size(2)))
            e1_rep = e1_rep.reshape(e1_rep.size(0), -1)
            e2_rep = e2_rep.reshape(e2_rep.size(0), -1)
            c_rep = torch.cat([e1_rep, e2_rep], dim=-1)

        elif self.args.af == 'avg':
            e1_len = e1_mask.sum(dim=-1).unsqueeze(-1)
            e2_len = e2_mask.sum(dim=-1).unsqueeze(-1)
            e1_rep = (e1_mask.unsqueeze(-1)*outputs).sum(dim=1)/e1_len
            e2_rep = (e2_mask.unsqueeze(-1)*outputs).sum(dim=1)/e2_len
            c_rep = torch.cat([e1_rep, e2_rep], dim=-1)

        elif self.args.af == 'max':
            e1_rep = e1_mask.unsqueeze(-1)*outputs
            e2_rep = e2_mask.unsqueeze(-1)*outputs
            e1_rep = e1_rep.max(dim=1)[0]
            e2_rep = e2_rep.max(dim=1)[0]
            c_rep = torch.cat([e1_rep, e2_rep], dim=-1)

        return self.out(c_rep)
