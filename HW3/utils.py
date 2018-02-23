# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
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

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

BOS_WORD = '<s>'
EOS_WORD = '</s>'
CLIP = 10
USE_CUDA = True if torch.cuda.is_available() else False

def escape(l):
	return l.replace("\"", "<quote>").replace(",", "<comma>")

def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]

def process_batch(batch):
    x, y = batch.src, batch.trg
    if USE_CUDA:
        x, y = x.cuda(), y.cuda()
    return x, y

def train_batch(model, source, target, optimizer, criterion):
    loss = 0
    model.zero_grad()
    output, hidden = model(source, target)
    output_flat = output.view(-1, model.output_size) # [(tg_len x batch) x en_vocab_sz]
    # not sure whether to use ground truth target or network's prediction
    loss = criterion(output_flat, target.view(-1)) # why is this true. what's decoder output
    loss.backward()

    # figure out how to do this
    # if L2 Norm of Gradient / 128 > 5, then g = 5g/s
    nn.utils.clip_grad_norm(model.parameters(), CLIP)
    optimizer.step()
    return loss.data[0]

def train(model, train_iter, epochs, optimizer, criterion, scheduler=None): # do I need a max_length=MAX_LENGTH?
    model.train()
    plot_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_iter:
            source, target = process_batch(batch)
            batch_loss = train_batch(model, source, target, optimizer, criterion)
            total_loss += batch_loss
        print(str(epoch) + "EPOCH LOSS: " + str(total_loss))

        if scheduler:
            scheduler.step()
        plot_losses.append(total_loss)
        # plot_losses_graph.append(plot_loss_avg)
    return plot_losses

# def evaluate(s, encoder, decoder, max_length): # need max_length?
#     input_var = s #somehow get input far from s

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

