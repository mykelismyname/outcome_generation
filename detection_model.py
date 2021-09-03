import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, batch_size, hdim, token_tags, token_label_embeddings):
        super(outcome_detection_model, self).__init__()
        self.batch_size = batch_size
        self.hdim = hdim
        self.token_tags = token_tags
        self.token_label_embeddings = token_label_embeddings

    def forward(self, x):
        pass