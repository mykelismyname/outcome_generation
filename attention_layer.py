import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import prepare_data

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
        combined_proj = torch.tanh(sum([inp_proj, pos_proj]))
        scores = self.U(combined_proj)
        attention_scores = F.softmax(scores, dim=0)
        return attention_scores


def fetch_prompt_conditioned_hidden_states(batch_prompt_types, batch_input_ids, batch_hidd_states, tokenizer, pos_embddings, seq_length, hdim, pos_attn, map_pos_ids):
    prompt_conditioned_outputs = []
    for ins in range(batch_input_ids.size(0)):
        if batch_prompt_types[ins] != 'null':
            pos_ids, single_mask_prompt = prepare_data.create_position_ids(input_ids=batch_input_ids[ins], template=batch_prompt_types[ins], tokenizer=tokenizer)
            if single_mask_prompt:
                pos_ids = [map_pos_ids[i] for i in pos_ids]
                pos_ids = torch.tensor(pos_ids)
                pos_embs = pos_embddings.index_select(0, pos_ids.to(device))
                pos_aware_attn = pos_attn(input_h_states=batch_hidd_states[ins], pos_features=pos_embs)
                pos_aware_attn, hidd_states = pos_aware_attn.unsqueeze(1), batch_hidd_states[ins].unsqueeze(-1)
                outputs = torch.bmm(hidd_states, pos_aware_attn).view(seq_length, hdim)
            else:
                outputs_ = []
                for ids in pos_ids:
                    pos_ids = torch.tensor(ids)
                    pos_embs = pos_embddings.index_select(0, pos_ids.to(device))
                    pos_aware_attn = pos_attn(input_h_states=batch_hidd_states[ins], pos_features=pos_embs)
                    pos_aware_attn, hidd_states = pos_aware_attn.unsqueeze(1), batch_hidd_states[ins].unsqueeze(-1)
                    output = torch.bmm(hidd_states, pos_aware_attn).view(seq_length, hdim)
                    outputs_.append(output)
                outputs = torch.mean(torch.stack(outputs_))
        else:
            outputs = batch_hidd_states[ins]
        prompt_conditioned_outputs.append(outputs)
    prompt_conditioned_outputs = torch.stack(prompt_conditioned_outputs)
    return prompt_conditioned_outputs