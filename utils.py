
import torch.nn as nn


""" init weights with xavier initialization """
def init_xavier_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)


""" learning-rate decay """
def lr_decay(lr: float=0, epoch: int=0, decay_rate: float=0.1, period: int=5) -> float:
    if epoch % period == 0:
        return lr * (1 / (decay_rate * epoch + 1))
    else:
        return lr