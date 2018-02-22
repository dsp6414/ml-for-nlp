import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import math
import numpy
import pdb
import spacy

import model

torch.manual_seed(1)

BOS_WORD = '<s>'
EOS_WORD = '</s>'
CLIP = 5
USE_CUDA = True if torch.cuda.is_available() else False

def escape(l):
	return l.replace("\"", "<quote>").replace(",", "<comma>")

def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]

def train(x, y, encoder, decoder, encoder_optm, decoder_optm, criterion): # do I need a max_length=MAX_LENGTH?
    loss = 0
    
    encoder_optm.zero_grad()
    decoder_optm.zero_grad()

    x_length = x.size()[0]
    y_length = y.size()[0]

    encoder_hidden = encoder.init_hidden()
    encoder_output, encoder_hidden = encoder(x, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[BOS_WORD]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden # last hidden state from encoder for start of decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    for i in range(y_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attn = \
            decoder(decoder_input, decoder_context, decoder_hidden, encoder_output)


        # not sure whether to use ground truth target or network's prediction
        loss += criterion(decoder_output[0], y[i]) # why is this true. what's decoder output
        decoder_input = y[i]
        
    loss.backward()
    nn.utils.clip_grad_norm(encoder.parameters(), CLIP)
    nn.utils.clip_grad_norm(decoder.parameters(), CLIP)
    encoder_optm.step()
    decoder_optm.step()

    return loss.data[0] / target_length

# def evaluate(s, encoder, decoder, max_length): # need max_length?
#     input_var = s # somehow get input far from s

#     encoder_hidden = encoder.init_hidden()
#     encoder_output, encoder_hidden = encoder(input_var, encoder_hidden)

#     decoder_input = Variable(torch.LongTensor([[BOS_WORD]])) # SOS
#     decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
#     if USE_CUDA:
#         decoder_input = decoder_input.cuda()
#         decoder_context = decoder_context.cuda()

#     decoder_hidden = encoder_hidden
#     decoder_attn = torch.zeros(max_length, max_length) # this is where I need max_length?
#     decoded = []

#     for i in range(max_length):
#         decoder_output, decoder_context, decoder_hidden, decoder_attn = \
#             decoder(decoder_input, decoder_context, decoder_hidden, encoder_output)

#         decoder_attn[i, :decoder_attn.size(2)] += decoder_attn.squeeze(0).squeeze(0).data # unsure what this does

#         # Figure out how to use decoder


# def plot_attention(s, encoder, decoder, max_length):
#     output_words, attn = evaluate(s, encoder, decoder, max_length)
#     print('input =', s)
#     print('output =', ' '.join(output_words))

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)

#     ax.set_xticklabels([''] + s.split(' ') + [EOS_WORD], rotation=90)
#     ax.set_yticklabels([''] + output_words)

#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.show()
#     plt.close()

