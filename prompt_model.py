import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import attention_layer as pos_attn
import prepare_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, model_path):
        checkpt = torch.load(model_path)
        self.load_state_dict(checkpt['model'])

    def save(self, epoch, path=None):
        params = {'model':self.state_dict(),
                  'epoch':epoch}
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(params, path)
        return path


class prompt_model(BasicModule):
    def __init__(self, model, special_token_ids, hdim, tokenizer, seq_length, add_marker_tokens, marker_token_emb_size, ner_label_ids, detection, prompt_conditioning, marker_tokens_not_trainable=False):
        super(prompt_model, self).__init__()
        self.model = model
        self.special_token_ids = special_token_ids
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.maker_token_emb_size = marker_token_emb_size
        self.vocab = self.tokenizer.get_vocab()
        self.hdim = hdim + self.maker_token_emb_size if add_marker_tokens else hdim
        self.fcl = nn.Linear(self.hdim, self.hdim)
        self.norm = nn.LayerNorm(self.hdim, eps=1e-12)
        self.output = nn.Linear(self.hdim, len(self.vocab), bias=False)
        self.bias = nn.Parameter(torch.zeros(len(self.vocab)))
        self.output.bias = self.bias
        self.linear_first = nn.Linear(self.hdim, 200)
        self.linear_second = nn.Parameter(torch.rand(200), requires_grad=True)

        if add_marker_tokens:
            self.add_marker_tokens = add_marker_tokens
            self.marker_tokens = nn.Embedding(len(self.special_token_ids), self.maker_token_emb_size)
        if marker_tokens_not_trainable:
            self.marker_tokens.weight.requires_grad = False

        if detection:
            self.detection = detection
            self.ner_output = nn.Linear(self.hdim, len(ner_label_ids))

        if prompt_conditioning:
            self.prompt_conditioning = prompt_conditioning
            self.pos_attn = pos_attn.position_attention(self.hdim, 200, 300).to(device)
            self.pos_emb = nn.Embedding(seq_length*2, 300)
            self.map_pos_ids = prepare_data.map_pos_neg_ids(self.seq_length)


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
            marker_token_tensors = self.marker_tokens.weight.data[x].repeat(seq_length, 1)
            h = torch.cat([marker_token_tensors, batch_hidden_output[i]], dim=1)
            marker_token_hidden[i] = h
        return marker_token_hidden

    def forward(self, input_ids, attention_mask, label_ids, layer='last'):
        batch_prompt_types = self.prompt_type(input_ids)
        if self.add_marker_tokens:
            input_ids = torch.cat((input_ids[:, :1], input_ids[:, 2:]), axis=1)
            attention_mask = torch.cat((attention_mask[:, :1], attention_mask[:, 2:]), axis=1)
            label_ids = torch.cat((label_ids[:, :1], label_ids[:, 2:]), axis=1)
            hidd = self.prepare_embeddings(input_ids, attention_mask, label_ids, layer)
            hidd = self.add_marker_token_embeddings(batch_hidden_output=hidd, batch_prompt_types=batch_prompt_types).to(device)
        else:
            hidd = self.prepare_embeddings(input_ids, attention_mask, label_ids, layer)
        if self.prompt_conditioning:
            prompt_conditioned_outputs = pos_attn.fetch_prompt_conditioned_hidden_states(
                batch_prompt_types=batch_prompt_types, batch_input_ids=input_ids,
                batch_hidd_states=hidd, tokenizer=self.tokenizer,
                pos_embddings=self.pos_emb.weight.data, seq_length=self.seq_length,
                hdim=self.hdim, pos_attn=self.pos_attn, map_pos_ids=self.map_pos_ids)
            if self.prompt_conditioning == 1:
                hidd = prompt_conditioned_outputs
            elif self.prompt_conditioning == 2:
                hidd = torch.mean(torch.stack([hidd, prompt_conditioned_outputs]), dim=0)
        actv_hidd = F.gelu(self.fcl(hidd))
        norm_hidd = self.norm(actv_hidd)
        prompt_output = self.output(norm_hidd)
        if self.detection:
            ner_output = self.ner_output(norm_hidd)
            output = (hidd, prompt_output, ner_output)
        else:
            output = (hidd, prompt_output, None)
        return output

