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
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2))
    return layers


class G(nn.Module):
    def __init__(self, opt):
        super(G, self).__init__()
        len_out = opt.channels * opt.img_size * opt.img_size
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.g = nn.Sequential(
            *block_G(opt.len_z + opt.n_classes, 128, bn=False),
            *block_G(128, 256),
            *block_G(256, 512),
            *block_G(512, 1024),
            nn.Linear(1024, len_out),
            nn.Tanh()
        )
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)


    def forward(self, z, labels):
        z = flow.cat((z, self.label_emb(labels)), -1)
        img = self.g(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class D(nn.Module):
    def __init__(self, opt):
        super(D, self).__init__()
        len_out = opt.channels * opt.img_size * opt.img_size
        self.d = nn.Sequential(
            nn.Linear(len_out+opt.n_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2,),
            nn.Linear(512, 1),
        )
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

    def forward(self, x, label):
        x_flat = flow.cat((x.view(x.size(0), -1), self.label_emb(label)), -1)
        return self.d(x_flat)


def sample_image(n_row, batches_done, g, one_hot):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = flow.tensor(np.random.normal(0, 1, (n_row ** 2, opt.len_z)), dtype=flow.float32)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = flow.tensor(labels)
    if flow.cuda.is_available():
        z, labels = z.cuda(), labels.cuda()
    g.eval()
    gen_imgs = g(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


def Train(opt, dataloader):
    g = G(opt).cuda() if flow.cuda.is_available() else G(opt)
    d = D(opt).cuda() if flow.cuda.is_available() else D(opt)

    adversarial_loss = flow.nn.MSELoss()

    optim_G = flow.optim.Adam(g.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_D = flow.optim.Adam(d.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            g.train()
            real = flow.ones(imgs.size(0), 1, requires_grad=False)
            fake = flow.zeros(imgs.size(0), 1, requires_grad=False)
            z_noises = flow.tensor((np.random.normal(0, 1, (imgs.shape[0], opt.len_z))), dtype=flow.float32)
            z_labels = flow.tensor((np.random.randint(0, opt.n_classes, imgs.shape[0])), dtype=flow.int64)
            if flow.cuda.is_available():
                real_imgs = imgs.cuda()
                real = real.cuda()
                fake = fake.cuda()
                z_noises = z_noises.cuda()
                z_labels = z_labels.cuda().int()
                labels = labels.cuda().int()

            optim_G.zero_grad()
            gen_imgs = g(z_noises, z_labels)
            g_loss = adversarial_loss(d(gen_imgs, z_labels), real)
            g_loss.backward()
            optim_G.step()

            optim_D.zero_grad()
            real_loss = adversarial_loss(d(real_imgs, labels), real)
            fake_loss = adversarial_loss(d(gen_imgs.detach(), z_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(opt.n_classes, batches_done, g, one_hot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_z', type=int, default=100)
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_epochs", type=int, default=100, help="")
    parser.add_argument("--n_classes", type=int, default=10, help="")
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