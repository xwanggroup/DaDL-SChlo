#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import csv

class Discrimnator(nn.Module):
    def __init__(self):
        super(Discrimnator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=1024),
            nn.Tanh()
        )


    def forward(self, noises):
        fake_imgs = self.net(noises)
        return fake_imgs


class Trainer:
    def __init__(self, generator, gen_optim,
                 discriminator, dis_optim,
                 critic_iterations=5, gp_lambda=10,
                 print_every=100, device='gpu'):
        self.device = device

        self.G = generator.to(device)
        self.G_opt = gen_optim
        self.D = discriminator.to(device)
        self.D_opt = dis_optim
        self.critic_iterations = critic_iterations
        self.gp_lambda = gp_lambda
        self.print_every = print_every


        self.fixed_noise = 2 * (torch.rand(1, 128, device=device) - 0.5)

    def gradient_penalty(self, real_imgs, fake_imgs):
        batch_size = real_imgs.size(0)

        alpha = torch.rand(batch_size, 1,device=self.device).expand_as(real_imgs)
        interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
        interpolated.requires_grad_()

        prediction = self.D(interpolated)

        gradients = autograd.grad(outputs=prediction, inputs=interpolated, grad_outputs=torch.ones_like(prediction),
                                  create_graph=True, retain_graph=True)[0] # 网络对输入变量的求导

        gradients = gradients.view(batch_size, -1)

        # Avoid vanish
        epsilon = 1e-12  #ε
        L2norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + epsilon)
        gp = self.gp_lambda * ((L2norm - 1) ** 2).mean()

        return gp


    def g_loss_function(self, fake_imgs):
        g_loss = -self.D(fake_imgs).mean()
        return g_loss


    def d_loss_function(self, real_imgs, fake_imgs):
        gp = self.gradient_penalty(real_imgs=real_imgs, fake_imgs=fake_imgs)
        d_loss = -(self.D(real_imgs).mean() - self.D(fake_imgs).mean()) + gp
        return d_loss

    def train_single_epoch(self, dataloader):

        d_running_loss = 0
        g_running_loss = 0

        length = len(dataloader)

        for batch_idx, data in enumerate(dataloader, 0):

            real_imgs = data.to(self.device)

            for _ in range(self.critic_iterations):
                noise = (torch.rand(real_imgs.size(0), 128, device=self.device) - 0.5) * 2
                fake_imgs = self.G(noise)
                d_loss = self.d_loss_function(real_imgs, fake_imgs)

                self.D_opt.zero_grad()
                d_loss.backward()
                self.D_opt.step()

                d_running_loss += d_loss.item()

            noise = (torch.rand(real_imgs.size(0), 128, device=self.device) - 0.5) * 2
            fake_imgs = self.G(noise)
            g_loss = self.g_loss_function(fake_imgs)

            self.G_opt.zero_grad()
            g_loss.backward()
            self.G_opt.step()

            g_running_loss += g_loss.item()

            if (batch_idx + 1) % self.print_every == 0:
                print('batch:{}/{}, loss(avg.): generator:{}, discriminator:{}'
                      .format(batch_idx + 1,
                              length,
                              d_running_loss/(self.print_every * self.critic_iterations),
                              g_running_loss/self.print_every))

                d_running_loss = 0
                g_running_loss = 0


    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            print('Epoch:{}'.format(epoch + 1))
            self.train_single_epoch(dataloader)

            with torch.no_grad():
                if epoch + 1:
                    one_dimension = self.G(self.fixed_noise)
                    print(one_dimension.cpu())


class CSVDataset(Dataset):
    def __init__(self, dir,label):
        self.dir = dir
        allData = pd.read_csv(self.dir)
        x_y = allData.values
        x = []

        for index, i in enumerate(x_y):
            if i[0] == float(label):
                print(np.array(i)) 
                x.append(np.array(i[1:]))
        self.dataset = np.array(x, dtype=np.float32)
            # np.random.shuffle(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def train_dataloader(path, batch_size=4, num_workers=0,label=1):
    dataloader = DataLoader(
        CSVDataset(path,label),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader



batch_size = 128
dataloader = train_dataloader('./train_feature.csv',batch_size,num_workers=0,label=1)

generator = Generator()
discriminator = Discrimnator()

print(generator)
print(discriminator)

initial_lr = 5e-4 
betas = (0.9, 0.99)
g_optimizer = optim.Adam(generator.parameters(), lr=initial_lr, betas=betas)
d_optimizer = optim.Adam(discriminator.parameters(), lr=initial_lr, betas=betas)

epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(generator, g_optimizer, discriminator, d_optimizer, device=device)
trainer.train(dataloader, epochs)

name = '1'
torch.save(trainer.G.state_dict(), './gen_p_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_p_' + name + '.pt')


num = 200
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
print(generator)
state_dict = torch.load('./gen_p_1.pt')
generator.load_state_dict(state_dict)
f = open('gan_features.csv', 'w', encoding='utf-8', newline="")
csv_write = csv.writer(f)
for i in range(num):
    rand = 2 * (torch.rand(1, 128, device=device) - 0.5)
    one_dimension = generator(rand).cpu().detach().numpy()[0].tolist()
    one_dimension = [str(i) for i in one_dimension]
    csv_write.writerow(one_dimension)







