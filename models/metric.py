'''
@Author: Qin Yang
@Date: 2020-03-12 16:47:52
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-12 21:46:40
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['discriminator_loss', 'generator_loss']

# Hinge loss used for Discriminator


def discriminator_loss(real, fake):
    real_loss = torch.mean(F.relu(1.-real))
    fake_loss = torch.mean(F.relu(1.+fake))
    return real_loss+fake_loss


# Hinge loss used for Generator
def generator_loss(real, fake):
    fake_loss = -torch.mean(fake)
    return fake_loss
