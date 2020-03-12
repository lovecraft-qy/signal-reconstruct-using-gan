'''
@Author: Qin Yang
@Date: 2020-03-12 17:34:17
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-12 17:39:02
'''
from .models import netG, netD
from .models import discriminator_loss, generator_loss


if __name__ == "__main__":
    G = netG()
