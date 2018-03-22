import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_sz, hidden_sz, hidden_sz_2):
        super(VAE, self).__init__()

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.hidden_sz_2 = hidden_sz_2

        self.fc1 = nn.Linear(input_sz, hidden_sz)
        self.fc21 = nn.Linear(hidden_sz, hidden_sz_2)
        self.fc22 = nn.Linear(hidden_sz, hidden_sz_2)
        self.fc3 = nn.Linear(hidden_sz_2, hidden_sz)
        self.fc4 = nn.Linear(hidden_sz, input_sz)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1) # mu, logvar

    def reparametrize(self, mu, logvar):
        if self.train:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_sz))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

