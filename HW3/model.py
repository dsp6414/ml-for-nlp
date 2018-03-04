import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import math
import pdb
import heapq


torch.manual_seed(1)

BATCH_SIZE = 32
USE_CUDA = True if torch.cuda.is_available() else False
MAX_LEN = 20
BOS_EMBED = 2
EOS_EMBED = 3
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.init_param = 0.08

        self.bidirectional = False
        self.num_directions = 2 if self.bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size // self.num_directions, n_layers, dropout=dropout_p, bidirectional=self.bidirectional)

    def init_hidden(self, batch_size=BATCH_SIZE):
        if USE_CUDA:
            return (Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size // self.num_directions)).cuda(),
            Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size // self.num_directions)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size // self.num_directions)),
                    Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size // self.num_directions)))

    def forward(self, inputs, hidden):
        embedding = self.embedding(inputs)
        output, hidden = self.rnn(embedding, hidden) # [num_layers x batch x hidden]
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

    def init_hidden(self, batch_size=BATCH_SIZE):
        if USE_CUDA:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

    def forward(self, target, last_hidden, encoder_outputs):

        # ENCODER OUTPUT NOT USED HERE. 
        word_embeddings = self.embedding(target)# [seq_len x B x N]
        word_embeddings = self.dropout(word_embeddings)
        output, hidden = self.rnn(word_embeddings, last_hidden)

        output = self.out(output)
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.attn = True

        # inputs is the true values for the target sentence from previous time step
        # last_hidden is the bottleneck hidden from processing all of encoder
        # encoder_outputs is
    def forward(self, target, last_hidden, encoder_outputs):
        # check: target is (seq_len, batch, input_size)
        if len(target.size()) == 1:
            target = target.unsqueeze(0)
        word_embeddings = self.embedding(target) # [seq_len x B x E]
        decoder_outputs, hidden = self.rnn(word_embeddings, last_hidden) # [seq_len x B x H] , [L x B x H]
        scores = torch.bmm(encoder_outputs.transpose(0, 1), decoder_outputs.transpose(1, 2).transpose(0, 2)) 
        attn_weights = F.softmax(scores, dim=1) # [B x source_len x target_len]
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs.transpose(0, 1))
        output = self.out(torch.cat((decoder_outputs.transpose(0, 1), context), 2))

        output = output.transpose(0, 1).contiguous() # [Seq_len x B x en_vocab]

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, n_layers=1, dropout=0.0, attn=False, k=5):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.init_param = 0.08
        self.attn = attn
        self.n_layers = n_layers
        self.encoder = EncoderRNN(input_size, embedding_size, hidden_size, n_layers, dropout)

        if attn:
            self.decoder = AttnDecoderRNN(embedding_size, hidden_size, output_size, n_layers, dropout)
        else:
            self.decoder = DecoderRNN(embedding_size, hidden_size, output_size, n_layers, dropout)

        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.valid = False
        print(self.attn)

    def forward(self, source, target, use_target=True, k=None):
        batch_size = len(source[1])
        encoder_hidden = self.encoder.init_hidden(batch_size=batch_size)
        encoder_outputs, encoder_hidden = self.encoder(source, encoder_hidden)

        decoder_hidden = tuple([e.view(self.n_layers, e.size(1), -1) for e in encoder_hidden]) # [num_layers x batch x hidden]. 

        # THE REAL KAGGLE THING
        if self.valid:
            decoder_hidden = encoder_hidden 
            initial_guess = Variable(torch.LongTensor([2]).view(1, 1))
            if USE_CUDA:
                initial_guess = initial_guess.cuda()
            current_hypotheses = [(0, initial_guess, decoder_hidden)]

            completed_guesses = []

            output_length = MAX_LEN

            for i in range(output_length):
                guesses_for_this_length = []
                while (current_hypotheses != []):
                    # Pop something off the current hypotheses
                    hypothesis = current_hypotheses.pop(0)
                    log_prob, last_sequence_guess, decoder_hidden = hypothesis
                    
                    last_word = last_sequence_guess[-1:, :]
                    # EOS token:
                    if last_word.squeeze().data[0] == 3: 
                        completed_guesses.append((log_prob, last_sequence_guess, None))
                    else:
                        if self.attn:
                            decoder_outputs, decoder_hidden, attn_weights = self.decoder(last_word, decoder_hidden, encoder_outputs)
                        else:
                            decoder_outputs, decoder_hidden = self.decoder(last_word, decoder_hidden, encoder_outputs)
                        # Get k hypotheses for each 
                        # decoder outputs is [target_len x batch x en_vocab_sz] -> [1 x 1 x vocab]
                        vocab_size = len(decoder_outputs[0][0])
                        n_probs, n_indices = torch.topk(decoder_outputs, k, dim=2)
                        new_probs = F.log_softmax(n_probs, dim=2) + log_prob# this should be tensor of size k 
                        new_probs = new_probs.squeeze().data
                        new_sequences = [torch.cat([last_sequence_guess, n_index.view(1, 1)],dim=0) for n_index in n_indices.squeeze()] # check this
                        new_hidden = [decoder_hidden] * k
                        # decoder_hidden: # tuple, each of which is [num_layers x batch x hidden]
                        seq_w_probs = list(zip(new_probs, new_sequences, new_hidden))
                        guesses_for_this_length = guesses_for_this_length + seq_w_probs

                # Top k current hypotheses after this time step:
                guesses_for_this_length = sorted(guesses_for_this_length, key= lambda tup: -1*tup[0])[:k]

                current_hypotheses = current_hypotheses + guesses_for_this_length

            # Return top result
            completed_guesses = completed_guesses + guesses_for_this_length

            completed_guesses.sort(key= lambda tup: -1*tup[0])
            return [x[1] for x in completed_guesses]

        # TRAINING AND VALIDATION:
        if self.attn:
            decoder_outputs, decoder_hidden, attn_weights = self.decoder(target, decoder_hidden, encoder_outputs)
            return decoder_outputs, decoder_hidden, attn_weights
        else:
            decoder_outputs, decoder_hidden = self.decoder(target, decoder_hidden, encoder_outputs)
            return decoder_outputs, decoder_hidden # decoder_output [target_len x batch x en_vocab_sz]

