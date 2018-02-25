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

BATCH_SIZE = 128
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

    def init_hidden(self, batch_size=BATCH_SIZE):
        if USE_CUDA:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

    def forward(self, inputs, hidden):
        # seq_len = len(inputs)
        # embedding = self.embedding(inputs).view(seq_len, 1, -1) # check sizes here
        embedding = self.embedding(inputs) # [len x B x E]
        try:
            output, hidden = self.rnn(embedding, hidden) # [num_layers x batch x hidden]
        except:
            print("INPUTS")
            print(inputs)
            print("EMBEDDING")
            print(embedding)
            print("HIDDEN")
            print(hidden)
            print("RNN")
            print(self.rnn.parameters())
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, last_hidden, encoder_output):
        word_embedding = self.embedding(inputs).unsqueeze(0) # [1 x B x N]
        output, hidden = self.rnn(word_embedding, last_hidden)
        # output: [1 x batch x hidden]
        # hidden: [num_layer x batch x hidden], [num_layer x batch x hidden]
        output = output.squeeze(0) # check dim
        output = self.out(output)
        return output, hidden

class DecoderBeam(nn.Module):
    def __init__(self, decoder, k):
        super(DecoderRNN, self).__init__()
        self.decoder = decoder
        self.k = k 

    def forward(self, inputs, last_hidden, encoder_output, max_length):
        for i in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_output)
            # Get the k best predictions for each 
            for j in range()
        return decoder_output, decoder_hidden

class AttnNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(AttnNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size) # do I need first variable to be hidden_size * 2??

    def forward(self, hidden, encoder_output):
        h = hidden.repeat(timestep, 1, 1) # .transpose(0, 1) maybe don't need transpose
        attn_energies = torch.bmm(h, encoder_output)
        energy = hidden.dot(self.attn(encoder_output))

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.attn = AttnNetwork(hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, last_hidden, encoder_output):
        word_embedding = self.dropout(self.embedding(inputs).unsqueeze(0)) # [1 x B x N]

        attn_weights = self.attn(last_hidden[-1], encoder_output)
        context = attn_weights.bmm(encoder_output.transpose(0, 1)) # [B x 1 x N]

        combined = torch.cat((word_embedding, context), 2) # check this dimension
        output, hidden = self.rnn(combined, last_hidden)
        output = output.squeeze(0) # B x N (check dimensions)
        output = self.out(torch.cat((output, context), 1))

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, n_layers=1, dropout=0.0, attn=False):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.init_param = 0.08
        self.attn = attn
        self.encoder = EncoderRNN(input_size, embedding_size, hidden_size, n_layers, dropout)

        if attn:
            self.decoder = AttnDecoderRNN(embedding_size, hidden_size, output_size, n_layers, dropout)
        else:
            self.decoder = DecoderRNN(embedding_size, hidden_size, output_size, n_layers, dropout)
        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, source, target, use_target=False):
        max_length = len(target)
        batch_size = len(source[1])
        encoder_hidden = self.encoder.init_hidden(batch_size=batch_size) # can insert batch size here
        encoder_output, encoder_hidden = self.encoder(source, encoder_hidden)
        # encoder_output: [source_len x batch x hidden]
        # encoder_hidden: # [num_layers x batch x hidden]
        decoder_outputs = Variable(torch.zeros(max_length, batch_size, self.output_size))
        if USE_CUDA:
            decoder_outputs = decoder_outputs.cuda()

        # decoder_input = Variable(torch.LongTensor([[BOS_WORD]]))
        decoder_output = Variable(target[0].data) # [1 x batch]
        decoder_hidden = encoder_hidden # [num_layers x batch x hidden]
        # decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        if USE_CUDA:
            decoder_output = decoder_output.cuda()
            # decoder_context = decoder_context.cuda()

        def beam_search(k):
            for i in range(0, max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_output)
                # decoder_output: [batch x len(EN)]
                # target: [target_len x batch]
                # For each word, keep the "k" best guesses
                pdb.set_trace()
                values, indices = torch.sort(decoder_output, dim = 1, descending=True)
                
            return decoder_output, decoder_hidden

        return beam_search(5)


        for i in range(0, max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_output)
            # decoder_output: [batch x len(EN)]
            # target: [target_len x batch]
            decoder_outputs[i] = decoder_output
            pdb.set_trace()
            if use_target:
                decoder_output = target[i].cuda() if USE_CUDA else target[i]
            else:
                decoder_output = decoder_output.max(1)[1]
        # decoder_output, hidden = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_output)
        return decoder_outputs, decoder_hidden # decoder_output [target_len x batch x en_vocab_sz]


