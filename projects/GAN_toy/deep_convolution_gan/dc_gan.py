import oneflow as flow
import oneflow.nn as nn
import argparse
import numpy as np

import flowvision.datasets
import flowvision.transforms as transforms
from flowvision import datasets
from projects.GAN_toy.gan_util.util import save_image
import sys


class G(nn.Module):
    def __init__(self,):
        super(G, self).__init__()
        self.g = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                nn.init.constant_(m.bias.data, 0)


    def forward(self, z):
        img = self.g(z)
        return img


class D(nn.Module):
    def block(self, in_channels, out_channels, bn=False):
        block_ = [
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d()
        ]
        if bn:
            block_.append(nn.BatchNorm2d(out_channels))
        return block_


    def __init__(self,):
        super(D, self).__init__()
        self.d = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.d(img)

def Train(opt, dataloader):
    g = G().cuda() if flow.cuda.is_available() else G()
    d = D().cuda() if flow.cuda.is_available() else D()

    adversarial_loss = flow.nn.BCELoss()

    optim_G = flow.optim.Adam(g.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_D = flow.optim.Adam(d.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real = flow.ones(imgs.size(0), 1, 1, 1, requires_grad=False)
            fake = flow.zeros(imgs.size(0), 1, 1, 1, requires_grad=False)
            z = flow.Tensor((np.random.normal(0, 1, (imgs.shape[0], opt.len_z, 1, 1))))
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
        flow.save({'g':g.state_dict(), 'd':d.state_dict()}, './g_d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_z', type=int, default=100)
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_epochs", type=int, default=100, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    opt = parser.parse_args()
    print(opt)

    dataloader = flow.utils.data.DataLoader(
        datasets.ImageFolder(
            "/home/kaijie/Documents/datasets/gan/celeba", # path to your image
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                 transforms.CenterCrop(opt.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    Train(opt, dataloader)