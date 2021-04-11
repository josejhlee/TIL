import argparse
import os
#import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from trainers import Trainer
import models

"""
dataset_path
    celeba, cifar10
arch
    dcgan, resnet, stylegan, stylegan2
advloss
    vanilla, wasserstein, ls, hinge
reg
    wgangp, r1, sn
image_size
batch_size
ckpt_every
    number of iterations between saving ckpts
sample_every
    number of iterations between saving fake samples
eval_every
    number of iterations between evaluation
print_every
    number of iterations between stdouts
"""


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find('ConV') != -1 : # convolutional layer exist
        torh.nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1 : # batchnorm layer eixist
        torhc.nn.init.normal_(m.weight.data, 1.0, 0.2)
        torhc.nn.constant_(m.bias.data, 0)

if __name__ =='__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arch', type=str, default="DCGAN", help="network name. DCGAN,stylegan,stlyegan2")
    parser.add_argument('--advloss', type=str, default="vanilla", help="loss type. vanilla, wasserstain, ls, hinge")
    parser.add_argument('--reg', type=str, default=None, help="wgangp, r1, sn")
    parser.add_argument('--dataset_dir', type=str, default='/home/minjung/minjung/dataset/CELEBA/', help="training data dir")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--batches', type=int, default=100000, help ="specify number of batches to train")
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--log_folder', type=str, default="./log")
    parser.add_argument('--model_folder', type=str, default="./save_model")
    parser.add_argument('--image_folder', type=str, default="./save_img")
    args = parser.parse_args()

    for path in ([args.log_folder,args.model_folder,args.image_folder]):
        if not os.path.isdir(path):
            os.mkdir(path)


    transform = transforms.Compose([
    transforms.Scale(args.image_size),
    #transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(args.dataset_dir, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)



    generator = models.Generator()
    discriminator = models.Discriminator()
    
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    

    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)


    trainer = Trainer(data_loader,
                        args.print_every,
                        args.sample_every,
                        args.sample_num,
                        args.batches,
                        args.log_folder,
                        args.model_folder,
                        args.image_folder,
                        use_cuda=torch.cuda.is_available())

    trainer.train(generator, discriminator)




    