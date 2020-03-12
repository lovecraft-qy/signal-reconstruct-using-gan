'''
@Author: Qin Yang
@Date: 2020-03-12 18:00:01
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-13 00:13:29
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['norm_signal', 'down_sampling']


def norm_signal(x):
    mu = torch.mean(x, dim=-1, keepdim=True)
    sigma = torch.std(x, dim=-1, keepdim=True)
    return (x-mu)/sigma


def down_sampling(x, scale_factor=4):
    b, c, l = x.size()
    x = x.view(b, c, l // scale_factor, scale_factor)
    return x[:, :, :, 0].detach()
