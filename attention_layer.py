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
        self.U = nn.Linear(attn_size, hdim)


    def prepare_embeddings(self, input_ids, attention_mask, labels, layer):
        batch = transformers.BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})
        outputs = self.model(**batch, output_hidden_states=True)
        mlm_hidden_states = outputs.hidden_states
        # last layer
        if layer == 'last':
            emb = mlm_hidden_states[-1]
            return emb
        # average embeddings across all model layers
        elif layer == 'average':
            all_layer_hidd = [layer_hidd for layer_hidd in mlm_hidden_states[1:]]
            all_layer_hidd = torch.stack(all_layer_hidd, dim=0)
            emb = torch.mean(all_layer_hidd, dim=0)
            return emb

    def prompt_type(self, input_ids):
        batch_prompt_types = []
        special_ids_tokens = dict((v,k) for k,v in self.special_token_ids.items())
        for i in range(input_ids.size(0)):
            t = special_ids_tokens[input_ids[i][1].item()]
            batch_prompt_types.append(t.strip('[]'))
        return batch_prompt_types

    def prompt_conditioning(self, batch_hidden_output, batch_prompt_types):
        batch_size, seq_length, hidden_dimension = batch_hidden_output.size()
        batch_condition_hidden = torch.zeros(batch_size, seq_length, hidden_dimension)

        def compute_avg_represention(h):
            attn_in = torch.tanh(self.linear_first(h))
            attn_out = torch.matmul(attn_in, self.linear_second.unsqueeze(1))
            attn_dist = torch.softmax(attn_out, dim=0)
            cond_dist = torch.matmul(attn_dist.unsqueeze(1), h.unsqueeze(1))
            avg_cond_dist = torch.mean(cond_dist.view(-1, cond_dist.size(-1)), dim=0)
            print(h.shape, attn_in.shape, attn_out.shape, attn_dist.shape, cond_dist.shape, avg_cond_dist.shape, batch_hidden_output.shape)
            return avg_cond_dist

        for i in range(batch_hidden_output.size(0)):
            # print(1, batch_prompt_types[i])
            if batch_prompt_types[i] == 'prefix':
                for j in range(batch_hidden_output[i].size(0)):
                    h = batch_hidden_output[i][:j + 1]
                    print(batch_prompt_types[i])
                    avg_cond_dist = compute_avg_represention(h)
                    batch_condition_hidden[i][j] = avg_cond_dist
            elif batch_prompt_types[i] == 'postfix':
                for j in range(batch_hidden_output[i].size(0)):
                    h = batch_hidden_output[i][j:]
                    print(batch_prompt_types[i])
                    avg_cond_dist = compute_avg_represention(h)
                    batch_condition_hidden[i][j] = avg_cond_dist
            else:
                for j in range(batch_hidden_output[i].size(0)):
                    k = [m for m in range(batch_hidden_output[i].size(0)) if m != j]
                    context_indices = torch.tensor(k).to(device)
                    h = batch_hidden_output[i].index_select(0, context_indices)
                    print(batch_prompt_types[i])
                    avg_cond_dist = compute_avg_represention(h)
                    batch_condition_hidden[i][j] = avg_cond_dist

        return batch_condition_hidden

    def position_aware_attention(self):
        pass

    def add_marker_token_embeddings(self, batch_hidden_output, batch_prompt_types):
        # '[prefix]', '[postfix]', '[cloze]', '[mixed]', '[null]'
        batch_size, seq_length, hidden_dimension = batch_hidden_output.size()
        marker_token_hidden = torch.zeros(batch_size, seq_length, hidden_dimension+self.maker_token_emb_size)
        for i in range(batch_hidden_output.size(0)):
            if batch_prompt_types[i] == 'prefix':
                x = 0
            if batch_prompt_types[i] == 'postfix':
                x = 1
            if batch_prompt_types[i] == 'cloze':
                x = 2
            if batch_prompt_types[i] == 'mixed':
                x = 3
            if batch_prompt_types[i] == 'null':
                x = 4
            # print(batch_hidden_output[i].shape, self.marker_tokens.weight[x].shape)
            marker_token_tensors = self.marker_tokens.weight[x].repeat(seq_length, 1)
            h = torch.cat([marker_token_tensors, batch_hidden_output[i]], dim=1)
            # h = torch.add(batch_hidden_output[i], self.marker_tokens.weight[x].unsqueeze(0))
            # print(h.shape)
            marker_token_hidden[i] = h
        return marker_token_hidden

    def forward(self, input_ids, attention_mask, label_ids, layer='last', add_marker_tokens=False, detection=False, prompt_conditioning=None):
        if add_marker_tokens:
            batch_prompt_types = self.prompt_type(input_ids)
            input_ids = torch.cat((input_ids[:, :1], input_ids[:, 2:]), axis=1)
            attention_mask = torch.cat((attention_mask[:, :1], attention_mask[:, 2:]), axis=1)
            label_ids = torch.cat((label_ids[:, :1], label_ids[:, 2:]), axis=1)
            hidd = self.prepare_embeddings(input_ids, attention_mask, label_ids, layer)
            hidd = self.add_marker_token_embeddings(batch_hidden_output=hidd, batch_prompt_types=batch_prompt_types).to(device)
        else:
            hidd = self.prepare_embeddings(input_ids, attention_mask, label_ids, layer)
        if prompt_conditioning:
            if int(prompt_conditioning) == 1:
                hidd = self.prompt_conditioning(batch_hidden_output=hidd, batch_prompt_types=batch_prompt_types).to(device)
            elif int(prompt_conditioning) == 2:
                batch_prompt_types = self.prompt_type(input_ids)
                cond_hidd = self.prompt_conditioning(batch_hidden_output=hidd, batch_prompt_types=batch_prompt_types).to(device)
                hidd = torch.mean(hidd, cond_hidd)
        actv_hidd = F.gelu(self.fcl(hidd))
        norm_hidd = self.norm(actv_hidd)
        prompt_output = self.output(norm_hidd)
        if detection:
            ner_output = self.ner_output(norm_hidd)
            return hidd, prompt_output, ner_output
        else:
            return hidd, prompt_output, None

