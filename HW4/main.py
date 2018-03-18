import argparse
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
LR = 1e-3
SAMPLES = 64

train_dataset = datasets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = datasets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

parser = argparse.ArgumentParser(description='VAE MNIST')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


print(len(train_dataset))
print(len(test_dataset))
train_dataset[0][0]

torch.manual_seed(args.seed)
train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
train_label = torch.LongTensor([d[1] for d in train_dataset])
test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
test_label = torch.LongTensor([d[1] for d in test_dataset])
print(train_img[0])
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

model = model.VAE(img_width * img_height, HIDDEN1, HIDDEN2)
if USE_CUDA:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, args.epochs + 1):
    utils.train(model, train_loader, epoch, optimizer)
    utils.eval(model, val_loader, epoch)
    sample = Variable(torch.randn(SAMPLES, HIDDEN2))
    if USE_CUDA:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(SAMPLES, 1, img_width, img_height), 'results/sample_' + str(epoch) + '.png')
