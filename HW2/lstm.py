import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import math

import pdb

torch.manual_seed(1)

EMBEDDING_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 20

class LSTM(nn.Module):
    def __init__(self, embedding_size, vocab_size, num_layers=2, lstm_type='medium'):
        super(LSTM, self).__init__()
        self.lstm_type = lstm_type
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        if lstm_type == 'medium':
            self.hidden_size = 650
            self.init_param = 0.05
            self.dropout = 0.5
        elif lstm_type == 'large':
            self.hidden_size = 1500
            self.init_param = 0.04
            self.dropout = 0.65

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, self.hidden_size, num_layers, dropout=self.dropout)

        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.dropout = nn.Dropout(self.dropout)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.init_param, self.init_param)
        self.linear.weight.data.uniform_(-self.init_param, self.init_param)

    def init_hidden(self, batch_size=BATCH_SIZE):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(NUM_LAYERS, batch_size, self.hidden_size)).cuda(),
            Variable(torch.zeros(NUM_LAYERS, batch_size, self.hidden_size)).cuda())

        return (Variable(torch.zeros(NUM_LAYERS, batch_size, self.hidden_size)),
            Variable(torch.zeros(NUM_LAYERS, batch_size, self.hidden_size)))

    def forward(self, inputs, hidden):
        embedding = self.dropout(self.embedding(inputs)) # [bptt_len - 1 x batch x embedding_size]
        output, hidden = self.rnn(embedding, hidden) # [bptt_len - 1 x batch x units]
        output = self.dropout(output)
        output = self.linear(output.view(-1, self.hidden_size)) # [bptt_len - 1 x batch x vocab_size]
        return output, hidden

class LSTMExtension(nn.Module):
    def __init__(self, embedding_size, vocab_size, num_layers=2):
        super(LSTMExtension, self).__init__()
        self.embedding_size = 400
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.hidden_size = 1150
        self.init_param = 1.0 / math.sqrt(self.hidden_size)
        self.init_param_embedding = 0.01
        self.dropout = 0.3

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, self.hidden_size // 2, num_layers, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.dropout_embedding = nn.Dropout(0.1)
        self.dropout = nn.Dropout(self.dropout)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.init_param_embedding, self.init_param_embedding)
        self.linear.weight.data.uniform_(-self.init_param, self.init_param)

    def init_hidden(self, num_directions=2, batch_size=40):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.num_layers*num_directions, batch_size, self.hidden_size // num_directions)).cuda(),
            Variable(torch.zeros(self.num_layers*num_directions, batch_size, self.hidden_size // num_directions)).cuda())

        return (Variable(torch.zeros(self.num_layers*num_directions, batch_size, self.hidden_size // num_directions)),
            Variable(torch.zeros(self.num_layers*num_directions, batch_size, self.hidden_size // num_directions)))

    def set_weights(self, weights='weight_hh_l0', dropout=0.25):
        w = F.dropout(self.rnn.weight_hh_l0, p=dropout, training=True)

    def forward(self, inputs, hidden):
        self.set_weights()
        embedding = self.dropout_embedding(self.embedding(inputs)) # [bptt_len - 1 x batch x embedding_size]
        # hidden size: [layers x batch x units]
        output, hidden = self.rnn(embedding, hidden) # [bptt_len - 1 x batch x units]
        output = self.dropout(output)
        output = self.linear(output.view(-1, self.hidden_size)) # [bptt_len - 1 x batch x vocab_size]
        return output, hidden
