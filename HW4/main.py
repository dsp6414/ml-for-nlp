import argparse
import numpy as np
import torch
import torch.utils.data
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import model, utils
import pdb

USE_CUDA = True if torch.cuda.is_available() else False
HIDDEN1 = 400
HIDDEN2 = 20
SAMPLES = 64


parser = argparse.ArgumentParser(description='VAE MNIST')
parser.add_argument('--model', help='which model to use. VAE or GAN')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size for training (default: 128). For GAN: use 100 instead.')
parser.add_argument('--g-steps', type=int, help='how many steps of generator for 1 step of discriminator')
parser.add_argument('--d-steps', type=int, default=1, help='how many steps of discriminator')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--generator', help='default generator or pixel_cnn', default='default')
args = parser.parse_args()

torch.manual_seed(args.seed)

LR = 1e-3 if args.model =='VAE' else .0002

if args.model=='VAE':
    train_dataset = datasets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
    test_dataset = datasets.MNIST(root='./data/',train=False, 
                           transform=transforms.ToTensor())

    print(len(train_dataset))
    print(len(test_dataset))
    train_dataset[0][0]

    torch.manual_seed(args.seed)
    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])
    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])
    # print(train_img[0])
    print(train_img.size(), train_label.size(), test_img.size(), test_label.size())

    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()
    train_img = train_img[:10000]
    train_label = train_label[:10000]

    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)

    img_width = train_img.size()[2]
    img_height = train_img.size()[3]

elif args.model=='GAN':
    gan_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.MNIST(root='./data/',
                            train=True, 
                            transform=gan_transform,
                            download=True)
    test_dataset = datasets.MNIST(root='./data/',train=False, 
                           transform=gan_transform)


    train_img = torch.stack([torch.FloatTensor(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])

    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()
    train_img = train_img[:10000]
    train_label = train_label[:10000]

    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    img_width = train_img.size()[2]
    img_height = train_img.size()[3]

if args.model == 'VAE':
    LR = 1e-3
    model = model.ConditionalVAE(img_width * img_height, HIDDEN1, HIDDEN2)
    if USE_CUDA:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, args.epochs + 1):
        utils.train(model, train_loader, epoch, optimizer)
        utils.eval(model, val_loader, epoch)
        
        if epoch % 10 == 0:
	        sample = Variable(torch.randn(SAMPLES, HIDDEN2))
	        if USE_CUDA:
	            sample = sample.cuda()
	        
	        c = torch.zeros(SAMPLES).long().random_(0, 10).float()

	        sample = model.decode(sample, Variable(c)).cpu()
	        # pdb.set_trace()
	        save_image(sample.data.view(SAMPLES, 1, img_width, img_height), 'results/sample_meh_' + str(epoch) + '.png')

elif args.model == 'GAN':
    # Model params
    g_input_size = 100     # Random noise dimension coming into generator, per output vector
    g_hidden_size = 50   # Generator complexity
    g_output_size = img_width * img_height    # size of generated output vector
    d_input_size = 100   # Minibatch size - cardinality of distributions ???? Or is this img_width * img_height
    d_hidden_size = 50   # Discriminator complexity
    d_output_size = 1    # Single dimension for 'real' vs. 'fake'


    if args.generator == 'default':
        G = model.Generator(input_size=g_input_size, output_size=g_output_size)
    else:
        G = model.PixelCNN(input_size=g_input_size, output_size=g_output_size, img_width=img_width, img_height=img_height)
    D = model.Discriminator(input_size=img_width * img_height, output_size = d_output_size)

    if USE_CUDA:
        G.cuda()
        D.cuda()
    G_optimizer = optim.Adam(G.parameters(), lr=LR)
    D_optimizer = optim.Adam(D.parameters(), lr=LR)
    for epoch in range(1, args.epochs + 1):
        utils.train_minimax(D, G, train_loader, epoch, D_optimizer, G_optimizer, args.d_steps, args.g_steps, args.batch_size, g_input_size)
        # utils.eval_minimax(D, G, val_loader, epoch, args.batch_size)

    utils.gen_interpolated_examples(G, g_input_size)
