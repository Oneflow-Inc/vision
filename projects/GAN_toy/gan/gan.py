import oneflow as flow
import oneflow.nn as nn
import argparse
import numpy as np
import flowvision.transforms as transforms
from flowvision import datasets
from projects.GAN_toy.gan_util.util import save_image
import sys


def block_G(in_feat, out_feat, bn=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if bn:
        # layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.BatchNorm1d(out_feat,))
    layers.append(nn.LeakyReLU(0.2))
    return layers


class G(nn.Module):
    def __init__(self, opt):
        super(G, self).__init__()
        len_out = opt.channels * opt.img_size * opt.img_size
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.g = nn.Sequential(
            *block_G(opt.len_z, 128, bn=False),
            *block_G(128, 256),
            *block_G(256, 512),
            *block_G(512, 1024),
            nn.Linear(1024, len_out),
            nn.Tanh(),
        )


    def forward(self, z):
        img = self.g(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class D(nn.Module):
    def __init__(self, opt):
        super(D, self).__init__()
        len_out = opt.channels * opt.img_size * opt.img_size
        self.d = nn.Sequential(
            nn.Linear(len_out, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.d(x_flat)


def Train(opt, dataloader):
    g = G(opt).cuda() if flow.cuda.is_available() else G(opt)
    d = D(opt).cuda() if flow.cuda.is_available() else D(opt)


    adversarial_loss = flow.nn.BCELoss()

    optim_G = flow.optim.Adam(g.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_D = flow.optim.Adam(d.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real = flow.ones(imgs.size(0), 1, requires_grad=False)
            fake = flow.zeros(imgs.size(0), 1, requires_grad=False)
            z = flow.Tensor((np.random.normal(0, 1, (imgs.shape[0], opt.len_z))))
            if flow.cuda.is_available():
                real_imgs = imgs.cuda()
                real = real.cuda()
                fake = fake.cuda()
                z = z.cuda()

            optim_G.zero_grad()
            gen_imgs = g(z)
            g_loss = adversarial_loss(d(gen_imgs), real)
            g_loss.backward()
            optim_G.step()

            optim_D.zero_grad()
            real_loss = adversarial_loss(d(real_imgs), real)
            fake_loss = adversarial_loss(d(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], sys.path[0]+"/images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_z', type=int, default=100)
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_epochs", type=int, default=100, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    opt = parser.parse_args()
    print(opt)

    dataloader = flow.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    Train(opt, dataloader)