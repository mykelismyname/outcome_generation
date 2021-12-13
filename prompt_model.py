import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path

class prompt_model(BasicModule):
    def __init__(self, model, batch_size, hdim, tokenizer):
        super(prompt_model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.fcl = nn.Linear(hdim, hdim)
        self.norm = nn.LayerNorm(hdim, eps=1e-12)
        self.output = nn.Linear(hdim, len(self.vocab), bias=False)
        self.bias = nn.Parameter(torch.zeros(len(self.vocab)))
        self.output.bias = self.bias
        self.linear_first = nn.Linear(hdim, 200)
        self.linear_second = nn.Parameter(torch.rand(200), requires_grad=True)
        self.marker_tokens = nn.Embedding(len(self.tokenizer.all_special_tokens), hdim)

    def prepare_embeddings(self, input_ids, attention_mask, labels, layer):
        batch = transformers.BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})
        outputs = self.model(**batch, output_hidden_states=True)
        mlm_hidden_states = outputs.hidden_states
        # last layer
        if layer == 'last':
            emb = mlm_hidden_states[0]
            return emb
        # average embeddings across all model layers
        elif layer == 'average':
            all_layer_hidd = [layer_hidd for layer_hidd in mlm_hidden_states[1:]]
            all_layer_hidd = torch.stack(all_layer_hidd, dim=0)
            emb = torch.mean(all_layer_hidd, dim=0)
            return emb

    def prompt_type(self, input_ids):
        batch_prompt_types = []
        for i in range(input_ids.size(0)):
            t = self.tokenizer.convert_ids_to_tokens(ids=int(input_ids[i][1].item()))
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
            # print(h.shape, attn_in.shape, attn_out.shape, attn_dist.shape, cond_dist.shape, avg_cond_dist.shape, batch_hidden_output.shape)
            return avg_cond_dist
        
        for i in range(batch_hidden_output.size(0)):
            # print(1, batch_prompt_types[i])
            if batch_prompt_types[i] == 'prefix':
                for j in range(batch_hidden_output[i].size(0)):
                    h = batch_hidden_output[i][:j + 1]
                    avg_cond_dist = compute_avg_represention(h)
                    batch_condition_hidden[i][j] = avg_cond_dist
            elif batch_prompt_types[i] == 'postfix':
                for j in range(batch_hidden_output[i].size(0)):
                    h = batch_hidden_output[i][j:]
                    avg_cond_dist = compute_avg_represention(h)
                    batch_condition_hidden[i][j] = avg_cond_dist
            else:
                for j in range(batch_hidden_output[i].size(0)):
                    k = [m for m in range(batch_hidden_output[i].size(0)) if m != j]
                    context_indices = torch.tensor(k).to(device)
                    h = batch_hidden_output[i].index_select(0, context_indices)
                    avg_cond_dist = compute_avg_represention(h)
                    batch_condition_hidden[i][j] = avg_cond_dist
                    
        return batch_condition_hidden
    
    def add_marker_token_embeddings(self, batch_hidden_output, batch_prompt_types):
        #'[prefix]', '[postfix]', '[cloze]', '[mixed]', '[null]'
        batch_size, seq_length, hidden_dimension = batch_hidden_output.size()
        marker_token_hidden = torch.zeros(batch_size, seq_length+1, hidden_dimension)
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
            # print(batch_prompt_types[i], batch_hidden_output[i].shape, self.marker_tokens.weight[x].shape)
            h = torch.cat([self.marker_tokens.weight[x].unsqueeze(0), batch_hidden_output[i]], dim=0)
            # h = torch.add(batch_hidden_output[i], self.marker_tokens.weight[x].unsqueeze(0))
            # print(h.shape)
            marker_token_hidden[i] = h
        return marker_token_hidden

    def forward(self, input_ids, attention_mask, labels, layer='last', add_marker_tokens=False, prompt_conditioning=False):
        batch_prompt_types = self.prompt_type(input_ids)
        if add_marker_tokens:
            input_ids = torch.cat((input_ids[:, :1], input_ids[:, 2:]), axis=1)
            attention_mask = torch.cat((attention_mask[:, :1], attention_mask[:, 2:]), axis=1)
            labels = torch.cat((labels[:, :1], labels[:, 2:]), axis=1)
            hidd = self.prepare_embeddings(input_ids, attention_mask, labels, layer)
            hidd = self.add_marker_token_embeddings(batch_hidden_output=hidd, batch_prompt_types=batch_prompt_types).to(device)
        else:
            hidd = self.prepare_embeddings(input_ids, attention_mask, labels, layer)
        if prompt_conditioning:
            hidd = self.prompt_conditioning(batch_hidden_output=hidd, batch_prompt_types=batch_prompt_types).to(device)
        actv_hidd = F.gelu(self.fcl(hidd))
        norm_hidd = self.norm(actv_hidd)
        prompt_output = self.output(norm_hidd)
        # pt_output = torch.softmax(hid_output, dim=2)
        return hidd, prompt_output

