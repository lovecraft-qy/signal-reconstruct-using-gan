'''
@Author: Qin Yang
@Date: 2020-03-12 17:34:17
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-13 00:27:55
'''
import torch
import argparse
import torch.optim as optim
from models.cgan import Generator, Discriminator
from models.metric import discriminator_loss, generator_loss
from data.transform import norm_signal, down_sampling
from data.loader import gen_data_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--input_channels', default=12, type=int)
    parser.add_argument('--hidden_channels', default=128, type=int)
    parser.add_argument('--scale_factor', default=4, type=int)
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--lr_G', default=5e-4, type=float)
    parser.add_argument('--lr_D', default=1e-3, type=float)
    parser.add_argument('--betas', default=(0.9, 0.99), type=tuple)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--use_cuda', default=True)
    args = parser.parse_args()
    return args


def main():
    # Data loader
    opts = parse_args()
    data_path = opts.data_path
    batch_size = opts.batch_size
    num_workers = opts.num_workers
    data_loader = gen_data_loader(data_path, batch_size, num_workers)

    # Model options
    use_cuda = opts.use_cuda
    input_channels = opts.input_channels
    hidden_channels = opts.hidden_channels
    scale_factor = opts.scale_factor
    num_layers = opts.num_layers

    netG = Generator(c_dim=input_channels,
                     num_channels=hidden_channels,
                     scale_factor=scale_factor)
    netD = Discriminator(c_dim=input_channels,
                         num_layers=num_layers,
                         num_channels=hidden_channels)
    if use_cuda:
        netG.cuda()
        netD.cuda()

    # Optimizer option
    lr_G = opts.lr_G
    lr_D = opts.lr_D
    betas = opts.betas
    optimG = optim.Adam(netG.parameters(), lr=lr_G, betas=betas)
    optimD = optim.Adam(netD.parameters(), lr=lr_D, betas=betas)

    # Train the generator and discriminator
    epoch = opts.epoch
    num_samples = len(data_loader)
    num_batch = (num_samples//batch_size)+1

    for e in range(epoch):
        for i, (real_x, _)in enumerate(data_loader):
            # normalize the signal
            real_x = norm_signal(real_x).detach()
            cond_x = down_sampling(real_x, scale_factor)
            cond_x = norm_signal(cond_x).detach()

            # forward models
            fake_x = netG(cond_x)
            real = netD(real_x)
            fake = netD(fake_x)

            # train discriminator
            optimD.zero_grad()
            d_loss = discriminator_loss(real, fake)
            d_loss.backward()
            optimD.step()

            # train generator
            optimG.zero_grad()
            g_loss = generator_loss(real, fake)
            g_loss.backward()
            optimG.step()

            print("Epoch:[%d/%d], Batch:[%d/%d], GLoss:%0.6f,DLoss:%0.6f"
                  % (e+1, epoch, i+1, num_batch, g_loss.item(), d_loss.item()))


if __name__ == "__main__":
    main()
