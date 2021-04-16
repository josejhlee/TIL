import os
import time
import json
from tqdm import tqdm

import numpy as np


import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from torch.autograd import Variable
import torch.optim as optim

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Trainer(object):
    def __init__(self, data_sampler, log_interval, sample_interval, sample_num, train_batches, log_folder, model_folder, image_folder, use_cuda=False):

        self.data_sampler = data_sampler
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.sample_num = sample_num
        self.batch_size = self.data_sampler.batch_size
        self.train_batches = train_batches
        self.log_folder = log_folder
        self.model_folder = model_folder
        self.image_folder = image_folder
        self.use_cuda = use_cuda

        #self.gan_criterion = nn.BCEWithLogitsLoss() #without sigmoid layer in model
        self.gan_criterion = nn.BCELoss() #sigmoid layer in model
        self.image_enumerator = None

    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    def sample_real_batch(self):
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.data_sampler)

        batch_idx, batch = next(self.image_enumerator)

        b = batch[0]
        if self.use_cuda :
            b=b.cuda()

        if batch_idx == len(self.data_sampler) -1 :
            self.image_enumerator = enumerate(self.data_sampler)
        
        return b


    def train_discriminator(self, discriminator,sample_true, fake_batch, opt):
        #opt.zero_grad()


        real_batch = sample_true()
        batch = Variable(real_batch,requires_grad=False)
        real_labels = discriminator(batch)
        #real_labels = real_labels.view(-1)

        #fake_batch = sample_fake(self.batch_size)
        fake_labels = discriminator(fake_batch.detach())
        #fake_labels = fake_labels.view(-1)

        ones = self.ones_like(real_labels)
        zeros = self.zeros_like(fake_labels)


        l_discriminator = self.gan_criterion(real_labels, ones) + self.gan_criterion(fake_labels, zeros)
        l_discriminator.backward()
        opt.step()

        return l_discriminator

    def train_generator(self, discriminator, fake_batch, opt):
        #opt.zero_grad()

        #fake_batch = sample_fake(self.batch_size)
        fake_labels = discriminator(fake_batch)
        #fake_labels = fake_labels.view(-1)
        all_ones = self.ones_like(fake_labels)

        l_generator = self.gan_criterion(fake_labels, all_ones)
        l_generator.backward()
        opt.step()

        return l_generator

    def train(self, generator, discriminator):
        writer = SummaryWriter(log_dir=self.log_folder)

        if self.use_cuda:
            generator.cuda()
            discriminator.cuda()

        opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        opt_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

        def sample_fake(batch_size):
            return generator(torch.randn(batch_size, 100, 1, 1,device=torch.device("cuda:0" if self.use_cuda else "cpu")))

        def init_logs():
            return {'l_gen' : [], 'l_dis' : []}

        #batch_num = 0

        logs = init_logs()

        start_time = time.time()

        fixed_z = torch.randn(self.sample_num, 100, 1, 1,device=torch.device("cuda:0" if self.use_cuda else "cpu"))
        #while True :
        for batch_num in  tqdm(range(self.train_batches)) : 
            generator.train()
            discriminator.train()

            fake_batch = sample_fake(self.batch_size)
            discriminator.zero_grad()
            l_dis = self.train_discriminator(discriminator, self.sample_real_batch, fake_batch, opt_discriminator)
            generator.zero_grad()
            l_gen = self.train_generator(discriminator,fake_batch, opt_generator)

            logs['l_gen'].append(l_gen.data.item())
            logs['l_dis'].append(l_dis.data.item())

            #batch_num += 1
            #print("logs['l_gen'] : {}  ,  logs['l_dis'] : {}".format(l_gen.data.item(),l_dis.data.item()))



            if batch_num % self.log_interval == 0 :
                torch.save({'generator' : generator.state_dict(),
                            'discriminator' : discriminator.state_dict(),
                            'opt_generator' : opt_generator.state_dict(),
                            'opt_discriminator' : opt_discriminator.state_dict()},
                            os.path.join(os.path.join(self.model_folder, 'model_%05d.pytorch' % batch_num)))
                #torch.save(generator, os.path.join(self.model_folder, 'generator_%05d.pytorch' % batch_num))
                print("model save in :",os.path.join(self.model_folder, 'model_%05d.pytorch' % batch_num))
                
                writer.add_scalar('generator loss',l_gen.data.item(),batch_num)
                writer.add_scalar('discriminator loss',l_dis.data.item(),batch_num)
                
                with open(os.path.join(self.log_folder, 'logs_%05d.json' % batch_num), "w") as json_file:
                    json.dump(logs, json_file)


            if batch_num % self.sample_interval == 0:

                generator.eval()
                fake = generator(fixed_z)
                
                writer.add_image('fake_samples', make_grid(fake.data, normalize=True), batch_num)
                save_image(fake.data,os.path.join(self.image_folder,'batch_num_%d.png'%batch_num),normalize=True)
            
                """
                for i in range(fake.size()[0]):
                    img = fake[i]
                    save_image(img,os.path.join(save_path,'sample_%d.png' % i))
                """
        
        torch.save({'generator' : generator.state_dict(),
                    'discriminator' : discriminator.state_dict(),
                    'opt_generator' : opt_generator.state_dict(),
                    'opt_discriminator' : opt_discriminator.state_dict()},
                    os.path.join(os.path.join(self.model_folder, 'model_%05d.pytorch' % batch_num)))
        #torch.save(generator, os.path.join(self.model_folder, 'generator_%05d.pytorch' % batch_num))
        print("model save in :",os.path.join(self.model_folder, 'model_%05d.pytorch' % batch_num))
        
        writer.add_scalar('generator loss',l_gen.data.item(),batch_num)
        writer.add_scalar('discriminator loss',l_dis.data.item(),batch_num)
        
        with open(os.path.join(self.log_folder, 'logs_%05d.json' % batch_num), "w") as json_file:
            json.dump(logs, json_file)
        
        writer.close()
