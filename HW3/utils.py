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
import numpy as np
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
MAX_LEN = 20
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

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,-1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def train_batch(model, source, target, optimizer, criterion):
    loss = 0
    model.zero_grad()
    output, hidden = model(source, target, use_target=True)

    # here, I need to make sure that I'm not comparing the batch ground truch with <s> ... </s> with the predicted output
    # because then I'd be literally trying to match words up. I need to shift ground truth to be target[1:] and compare this to
    # output[:-1] <- output minus the last character

    output_flat = output.view(-1, model.output_size) # [(tg_len x batch) x en_vocab_sz]
    # not sure whether to use ground truth target or network's prediction
    loss = criterion(output_flat, target.view(-1))
    loss.backward()

    # figure out how to do this
    # if L2 Norm of Gradient / 128 > 5, then g = 5g/s
    nn.utils.clip_grad_norm(model.parameters(), CLIP)
    optimizer.step()

    # Subtract one so that the padding count becomes zero. Count number of nonpadding in output
    non_padding = (target.view(-1) - 1.0).nonzero().size(0)

    return loss.data[0], non_padding

    # keep track of number of non-padding tokens in each batch, and divide by that number

def train(model, train_iter, val_iter, epochs, optimizer, criterion, scheduler=None, filename=None):
    model.train()
    plot_losses = []
    counter = 0

    total_observations = 0

    stop_after_one_batch = False
    if epochs == 0:
        epochs = 1
        stop_after_one_batch = True

    for epoch in range(epochs):
        total_loss = 0
        counter = 0
        for batch in train_iter:
            source, target = process_batch(batch) # Source is 11x28, target is 21x28
            batch_loss, nonpadding = train_batch(model, source, target, optimizer, criterion)
            total_loss += batch_loss * non_padding
            total_observations += nonpadding

            if counter % 50 == 0:
                print("batch", str(counter), " : perplexity = ", np.exp((total_loss/total_observations)))
            if stop_after_one_batch:
                return plot_losses
            counter += 1

        print(str(epoch) + "EPOCH LOSS: " + str(total_loss), "PERPLEXITY:", np.exp((total_loss/total_observations)))

        if scheduler:
            scheduler.step()
        plot_losses.append(total_loss)

        print("Validate:", evaluate(model, val_iter, criterion))

        filename = 'seq2seq_2_25_' if filename is None else filename[:-4] 
        torch.save(model.state_dict(), filename + str(epoch) + '.sav')
        # plot_losses_graph.append(plot_loss_avg)
    return plot_losses

def evaluate(model, val_iter, criterion):
    model.eval()
    model.valid = True
    total_loss = 0.
    total_len = 0.
    for batch in val_iter:
        source, target = process_batch(batch)
        # output, hidden, metadata = model(source, target)
        if model.beam:
            output, hidden, metadata = model(source, target)
        else:
            output, hidden = model(source, target)
        output_flat = output.view(-1, model.output_size)
        loss = criterion(output_flat, target.view(-1))

        # - 1 hack for nonzero
        non_padding = (target.view(-1) - 1.0).nonzero().size(0)

        # Remove n
        total_loss += non_padding * loss.data
        total_len += non_padding

    print("Total Loss ", total_loss[0])
    print("Total Len ", total_len)
    print(total_loss[0] / total_len)
    model.train()
    model.valid = False
    return np.exp(total_loss / total_len), output

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
def kaggle(model, SRC_LANG, TRG_LANG,  output_file, input_file='source_test.txt'):
    pdb.set_trace()
    model.eval()
    model.valid = True
    f = open(input_file)
    lines = f.readlines()
    with open(output_file, 'w') as out:
        print('id,word', file=out)
        for i, line in enumerate(lines):
            text = Variable(torch.LongTensor([SRC_LANG.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1) # Shape: [len x 1]
            fake_target = Variable(torch.LongTensor([0] * MAX_LEN))
            if USE_CUDA:
                text = text.cuda()
                fake_target = fake_target.cuda()
            output, hidden, metadata = model(text, fake_target, k=100, use_target=False) # THE ONLY TIME USE_TARGET = FALSE
            sequences = torch.stack(metadata['topk_sequence']).squeeze() # should be max_len x k
            # convert each seq to sentence
            print("%d,", i, end='', file=out)
            for l in range(100):
                seq = sequences[:, l]
                english_seq = [TRG_LANG.vocab.itos[j.data[0]] for j in seq]

                # Only get first 3 ## DOUBLE CHECK thiS IS ACTUALLY FIRST NOT LAST LOL 
                english_seq = english_seq[:3]
                english_seq = escape("|".join(english_seq))
                print(english_seq, end= ' ',file=out)
            print(file=out)

    model.train()
    model.valid = False
