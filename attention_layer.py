import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class position_attention(nn.Module):
    def __init__(self, hdim, attn_size, pos_emb_size):
        super(position_attention, self).__init__()
        self.hdim = hdim
        self.attn_size = attn_size
        self.pos_emb_size = pos_emb_size

        self.W = nn.Linear(hdim, attn_size)
        self.V = nn.Linear(pos_emb_size, attn_size)
        self.U = nn.Linear(attn_size, 1)

    def forward(self, input_h_states, pos_features):
        inp_proj = self.W(input_h_states)
        pos_proj = self.V(pos_features)
        combined_proj = torch.tanh(torch.sum([inp_proj, pos_proj]))
        scores = self.U(combined_proj)
        attention_scores = F.softmax(scores, dim=1)
        output = torch.bmm(input_h_states, attention_scores)
        return output


