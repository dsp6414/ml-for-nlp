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
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.init_param = 0.08

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)

    def init_hidden(self, batch_size=128):
        if USE_CUDA:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda,
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

    def forward(self, inputs, hidden):
        # seq_len = len(inputs)
        # embedding = self.embedding(inputs).view(seq_len, 1, -1) # check sizes here
        embedding = self.embedding(inputs)
        output, hidden = self.rnn(embedding, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing

        # self.embedding = nn.Embedding(output_size, hidden_size)
        # self.attn = AttnNetwork(hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, last_hidden, encoder_output):
        # word_embedding = self.embedding(inputs).view(1, 1, -1) # [1 x B x N]
        # word_embedding = self.dropout(word_embedding)

        # attn_weights = self.attn(last_hidden[-1], encoder_output)
        # context = attn_weights.bmm(encoder_output.transpose(0, 1)) # [B x 1 x N]

        # combined = torch.cat((word_embedding, context), 2) # check this dimension
        # output, hidden = self.rnn(combined, last_hidden)

        # output = output.squeeze(0) # B x N (check dimensions)
        # output = self.out(torch.cat((output, context), 1))

        output, hidden = self.rnn(inputs, last_hidden)
        output = output.squeeze(0) # check dim
        output = self.out(output)

        return output, hidden
        # return output, hidden, attn_weights

# class AttentionNetwork(nn.Module):
#     def __init__(self, method, hidden_size, max_length=MAX_LENGTH): # check why we need max_length
#         super(AttentionNetwork, self).__init__()


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, n_layers=1, dropout=0.0):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.init_param = 0.08

        self.encoder = EncoderRNN(input_size, embedding_size, hidden_size, n_layers, dropout)
        self.decoder = DecoderRNN(embedding_size, hidden_size, output_size, n_layers, dropout)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.init_param, self.init_param)

    def forward(self, inputs):
        self.init_weights()
        max_length = len(inputs)

        encoder_hidden = (self.encoder.init_hidden(), self.encoder.init_hidden()) # can insert batch size here
        encoder_output, encoder_hidden = self.encoder(embedding, encoder_hidden)
        pdb.set_trace()
        decoder_outputs = Variable(torch.zeros(max_length, BATCH_SIZE, self.output_size))
        if USE_CUDA:
            decoder_outputs = decoder_outputs.cuda()

        decoder_input = Variable(torch.LongTensor([[BOS_WORD]]))
        decoder_hidden = encoder_hidden
        # decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            # decoder_context = decoder_context.cuda()

        for t in range(1, max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            decoder_outputs[t] = decoder_output
        # decoder_output, hidden = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_output)

        return decoder_outputs


