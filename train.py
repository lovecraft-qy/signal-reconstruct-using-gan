'''
@Author: Qin Yang
@Date: 2020-03-12 17:34:17
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-12 21:48:42
'''
import argparse
from models.cgan import netG, netD
from models.metric import discriminator_loss, generator_loss
from data.loader import gen_data_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr_G', default=0.0005, type=float)
    parser.add_argument('--lr_D', default=0.001, type=float)
    args = parser.parse_args()
    return args


def main():
    opts = parse_args()
    data_path = opts.data_path
    batch_size = opts.batch_size
    num_workers = opts.num_workers
    data_loader = gen_data_loader(data_path, batch_size, num_workers)

    for data, label in data_loader:
        print(label)


if __name__ == "__main__":
    main()
