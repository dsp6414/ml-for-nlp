import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import math
import pdb

import utils

torch.manual_seed(1)

USE_CUDA = True if torch.cuda.is_available() else False

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = 0.5 # need to check if this is a thing
        self.init_param = 0.08

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=self.dropout_p)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.init_param, self.init_param)
        self.rnn.weight.data.uniform_(-self.init_param, self.init_param)
        # self.linear.weight.data.uniform_(-self.init_param, self.init_param)

    def init_hidden(self):
        h = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA:
            h = h.cuda()
        return h

    def forward(self, inputs, hidden):
        self.init_weights()
        seq_len = len(inputs)
        embedded = self.embedding(inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = 0.5 # need to check if this is a thing

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = AttentionNetwork(hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, last_hidden, encoder_output):
        word_embedding = self.embedding(inputs).view(1, 1, -1) # [1 x B x N]
        word_embedding = self.dropout(word_embedding)

        attn_weights = self.attn(last_hidden[-1], encoder_output)
        context = attn_weights.bmm(encoder_output.transpose(0, 1)) # [B x 1 x N]

        combined = torch.cat((word_embedding, context), 2) # check this dimension
        output, hidden = self.rnn(combined, last_hidden)

        output = output.squeeze(0) # B x N (check dimensions)
        output = self.out(torch.cat((output, context), 1))

        return output, hidden, attn_weights

class AttentionNetwork(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH): # check why we need max_length
        super(AttentionNetwork, self).__init__()






