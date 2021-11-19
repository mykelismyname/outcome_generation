import torch
import torch.nn as nn
import torch.nn.functional as F
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

class outcome_detection_model(BasicModule):
    def __init__(self, batch_size, hdim, tokenizer, det_criterion, token_ids):
        super(outcome_detection_model, self).__init__()
        self.ner_output = nn.Linear(hdim, len(token_ids))
        self.tokenizer = tokenizer
        self.linear_first = nn.Linear(hdim, 200)
        self.linear_second = nn.Linear(200, len(token_ids))
        self.loss_dte = det_criterion

    def prepare_embeddings(self, all_layer_hidd, mode):
        # last layer
        if mode == 'last':
            emb = all_layer_hidd[-1]
            return emb
        #average embeddings across all model layers
        elif mode == 'average':
            all_layer_hidd = [layer_hidd for layer_hidd in all_layer_hidd[1:]]
            all_layer_hidd = torch.stack(all_layer_hidd, dim=0)
            all_layer_hidd = torch.mean(all_layer_hidd, dim=0)
            return all_layer_hidd

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

    def forward(self, input_ids, hidd_states, ner_labels, mode='last'):
        hidd = self.prepare_embeddings(hidd_states, mode)
        ner_batch_losses = []
        batch_prompt_types = self.prompt_type(input_ids)
        for i in range(hidd.size(0)):
            det_ouput = torch.empty([0]).to(device)
            if batch_prompt_types[i] == 'prefix':
                for j in range(hidd[i].size(0)):
                    h = hidd[i][:j+1]
                    attn = self.linear_first(h)
                    attn = torch.tanh(attn)
                    attn = self.linear_second(attn)
                    attn_dist = torch.bmm(attn.unsqueeze(-1), h.unsqueeze(1))
                    attn_dist = torch.mean(attn_dist, dim=2)
                    attn_dist = torch.mean(attn_dist, dim=0)
                    attn_dist = attn_dist.expand(1, attn_dist.size(0))
                    det_ouput = torch.cat([det_ouput, attn_dist])
            elif batch_prompt_types[i] == 'postfix':
                for j in range(hidd[i].size(0)):
                    h = hidd[i][j:]
                    attn = self.linear_first(h)
                    attn = torch.tanh(attn)
                    attn = self.linear_second(attn)
                    attn_dist = torch.bmm(attn.unsqueeze(-1), h.unsqueeze(1))
                    attn_dist = torch.mean(attn_dist, dim=2)
                    attn_dist = torch.mean(attn_dist, dim=0)
                    attn_dist = attn_dist.expand(1, attn_dist.size(0))
                    det_ouput = torch.cat([det_ouput, attn_dist])
            else:
                h = hidd[i]
                attn = self.linear_first(h)
                attn = torch.tanh(attn)
                attn = self.linear_second(attn)
                attn_dist = torch.bmm(attn.unsqueeze(-1), h.unsqueeze(1))
                attn_dist = torch.mean(attn_dist, dim=2)
                det_ouput = torch.cat([det_ouput, attn_dist])
            ner_preds = torch.softmax(det_ouput, dim=1)
            loss = self.comput_loss(ner_preds=ner_preds, ner_labels=ner_labels[i])
            ner_batch_losses.append(loss)
        # det_output = self.ner_output(h.squeeze(0))
        # ner_preds = torch.softmax(det_output, dim=2)
        return ner_batch_losses