import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

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
    def __init__(self, model, batch_size, hdim, tokenizer, det_criterion, ner_label_ids):
        super(prompt_model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.fcl = nn.Linear(hdim, hdim)
        self.norm = nn.LayerNorm(hdim, 1e-12)
        self.output = nn.Linear(hdim, len(self.vocab), bias=False)
        self.bias = nn.Parameter(torch.zeros(len(self.vocab)))
        self.output.bias = self.bias
        self.linear_first = nn.Linear(hdim, 200)
        self.linear_second = nn.Linear(200, len(ner_label_ids))
        self.loss_dte = det_criterion

    def prepare_embeddings(self, input_ids, attention_mask, labels, layer):
        batch = transformers.BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})
        outputs = self.model(**batch, output_hidden_states=True)
        mlm_loss = outputs.loss
        mlm_hidden_states = outputs.hidden_states
        # print(mlm_hidden_states.shape, outputs.logits.shape)
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

    def comput_loss(self, ner_preds, ner_labels):
        assert ner_preds.size(0) == ner_labels.size(0)
        ner_loss = self.loss_dte(ner_preds, ner_labels)
        return ner_loss

    def prompt_conditioning(self, batch_hidden_output, batch_prompt_types):
        batch_size, seq_length, hidden_dimension = batch_hidden_output.size()
        batch_condition_hidden = torch.zeros(batch_size, seq_length, hidden_dimension)
        for i in range(batch_hidden_output.size(0)):
            if batch_prompt_types[i] == 'prefix':
                for j in range(batch_hidden_output[i].size(0)):
                    h = batch_hidden_output[i][:j + 1]
                    attn = self.linear_first(h)
                    attn = torch.tanh(attn)
                    attn = self.linear_second(attn)
                    attn_dist = torch.bmm(attn.unsqueeze(-1), h.unsqueeze(1))
                    attn_dist = torch.mean(attn_dist, dim=2)
                    attn_dist = torch.mean(attn_dist, dim=0)
                    attn_dist = attn_dist.expand(1, attn_dist.size(0))
                    batch_condition_hidden[i][j] = attn_dist
            elif batch_prompt_types[i] == 'postfix':
                for j in range(batch_hidden_output[i].size(0)):
                    h = batch_hidden_output[i][j:]
                    attn = self.linear_first(h)
                    attn = torch.tanh(attn)
                    attn = self.linear_second(attn)
                    attn_dist = torch.bmm(attn.unsqueeze(-1), h.unsqueeze(1))
                    attn_dist = torch.mean(attn_dist, dim=2)
                    attn_dist = torch.mean(attn_dist, dim=0)
                    attn_dist = attn_dist.expand(1, attn_dist.size(0))
                    batch_condition_hidden[i][j] = attn_dist
            else:
                h = batch_hidden_output[i]
                attn = self.linear_first(h)
                attn = torch.tanh(attn)
                attn = self.linear_second(attn)
                attn_dist = torch.bmm(attn.unsqueeze(-1), h.unsqueeze(1))
                attn_dist = torch.mean(attn_dist, dim=2)

    def forward(self, input_ids, attention_mask, labels, ner_labels, layer='last'):
        hidd = self.prepare_embeddings(input_ids, attention_mask, labels, layer)
        batch_prompt_types = self.prompt_type(input_ids)
        hidd = F.gelu(self.fcl(hidd))
        hidd = self.norm(hidd)
        hid_output = self.output(hidd)
        # pt_output = torch.softmax(hid_output, dim=2)
        pt_output = hid_output
        return pt_output

