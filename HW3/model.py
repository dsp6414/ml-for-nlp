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
        # inputs: [seq_len x batch_sz]
        inverse_inputs = utils.flip(inputs, 0)
        embedding = self.embedding(inverse_inputs) # [len x B x E]
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

    def forward(self, inputs, last_hidden, encoder_outputs):
        word_embedding = self.embedding(inputs).unsqueeze(0) # [1 x B x N]
        output, hidden = self.rnn(word_embedding, last_hidden)
        # output: [1 x batch x hidden]
        # hidden: [num_layer x batch x hidden], [num_layer x batch x hidden]
        output = output.squeeze(0) # check dim
        output = self.out(output)
        return output, hidden

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        # for i in range(seq_len):
        #     attn_energies[i] = hidden.dot(self.attn(encoder_outputs[i]))

        attn_energies = torch.bmm(hidden.transpose(0, 1), encoder_outputs.transpose(0, 1).transpose(1, 2))

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=2)

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = Attn(hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # inputs is the true values for the target sentence from previous time step
        # last_hidden is the bottleneck hidden from processing all of encoder
        # encoder_outputs is
    def forward(self, inputs, last_hidden, encoder_outputs):
        word_embedding = self.dropout(self.embedding(inputs)) # [1 x B x Embedding]
        attn_weights = torch.bmm(last_hidden[0].transpose(0, 1), encoder_outputs.transpose(0, 1).transpose(1, 2))
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)
        combined = torch.cat((word_embedding, context), 1)
        #########################
        #  works up until here
        output, hidden = self.rnn(combined.unsqueeze(0), last_hidden)
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
        encoder_outputs, encoder_hidden = self.encoder(source, encoder_hidden)
        # encoder_outputs: [source_len x batch x hidden]
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
                decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_outputs)
                # decoder_output: [batch x len(EN)]
                # target: [target_len x batch]
                # For each word, keep the "k" best guesses
                pdb.set_trace()
                values, indices = torch.sort(decoder_output, dim = 1, descending=True)
                
            return decoder_output, decoder_hidden

        # return beam_search(5)


        for i in range(0, max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_outputs)
            # decoder_output: [batch x len(EN)]
            # target: [target_len x batch]
            decoder_outputs[i] = decoder_output
            if use_target:
                decoder_output = target[i].cuda() if USE_CUDA else target[i]
            else:
                decoder_output = decoder_output.max(1)[1]
        # decoder_output, hidden = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden # decoder_output [target_len x batch x en_vocab_sz]


