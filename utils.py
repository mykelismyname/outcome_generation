import sys
import torch

def save(model, config, epoch, path):
    params = {'model': model.state_dict(),
              'config':config,
              'epochs':epoch}
    try:
        torch.save(params, path)
    except Exception as e:
        print('Beware that no model has been saved, check the path provided')



