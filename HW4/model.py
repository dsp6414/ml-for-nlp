import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

class VAE(nn.Module):
    def __init__(self, input_sz, hidden_sz, hidden_sz_2):
        super(VAE, self).__init__()

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.hidden_sz_2 = hidden_sz_2

        self.input_to_hidden = nn.Linear(input_sz, hidden_sz)
        self.h1_to_h2 = nn.Linear(hidden_sz, hidden_sz_2)
        self.h1_to_h2_2 = nn.Linear(hidden_sz, hidden_sz_2)
        self.h2_to_h1 = nn.Linear(hidden_sz_2, hidden_sz)
        self.hidden_to_input = nn.Linear(hidden_sz, input_sz)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.input_to_hidden(x))
        return self.h1_to_h2(h1), self.h1_to_h2_2(h1) # mu, logvar

    def reparametrize(self, mu, logvar):
        if self.train:
            std = logvar.mul(0.5).exp_() # extract standard deviation
            eps = Variable(std.data.new(std.size()).normal_())
            return mu + std * eps #eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.h2_to_h1(z))
        return self.sigmoid(self.hidden_to_input(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_sz))
        pdb.set_trace()
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

class ConditionalVAE(nn.Module):
    def __init__(self, input_sz, hidden_sz, hidden_sz_2):
        super(ConditionalVAE, self).__init__()

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.hidden_sz_2 = hidden_sz_2

        self.input_to_hidden = nn.Linear(input_sz + 1, hidden_sz)
        self.h1_to_h2 = nn.Linear(hidden_sz, hidden_sz_2)
        self.h1_to_h2_2 = nn.Linear(hidden_sz, hidden_sz_2)
        self.h2_to_h1 = nn.Linear(hidden_sz_2 + 1, hidden_sz)
        self.hidden_to_input = nn.Linear(hidden_sz, input_sz)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.batch_sz = 128

    def encode(self, x, c):
        inputs = torch.cat((x, c), 1)
        h1 = self.relu(self.input_to_hidden(inputs))
        return self.h1_to_h2(h1), self.h1_to_h2_2(h1) # mu, logvar

    def reparametrize(self, mu, logvar):
        if self.train:
            std = logvar.mul(0.5).exp_() # extract standard deviation
            eps = Variable(std.data.new(std.size()).normal_())
            return mu + std * eps #eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat((z, c.unsqueeze(1)), 1)
        h3 = self.relu(self.h2_to_h1(inputs))
        return self.sigmoid(self.hidden_to_input(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.input_sz), c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar

class Generator(nn.Module):
    # initializers
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc4 = nn.Linear(self.fc2.out_features, output_size)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, output_size)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc4(x))

        return x

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self, input_size, output_size, img_width, img_height):
        """PixelCNN Model"""
        super(PixelCNN, self).__init__()
        fm = 64
        self.net = nn.Sequential(
            MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            nn.Conv2d(fm, 256, 1))
    
    def forward(self, x):
        pdb.set_trace()
        x = x.view(-1, 28, 28)
        out = self.net(x)
        return out

