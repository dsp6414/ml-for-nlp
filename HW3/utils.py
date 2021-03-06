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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def train_batch(model, source, target, optimizer, criterion, attn=False):
    loss = 0
    model.zero_grad()
    if attn:
        output, hidden, attention = model(source, target, use_target=True)
    else:
        output, hidden = model(source, target, use_target=True)

    # Take output minus the last character
    output = output[:-1, :, :]

    # Take target without the first character
    target = target[1:, :]

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

def train(model, train_iter, val_iter, epochs, optimizer, criterion, scheduler=None, filename=None, attn=False):
    model.train()
    plot_losses = []
    counter = 0

    stop_after_one_batch = False
    if epochs == 0:
        epochs = 1
        stop_after_one_batch = True

    for epoch in range(epochs):
        total_loss = 0
        counter = 0
        total_observations = 0
        for batch in train_iter:
            source, target = process_batch(batch) # Source is 11x28, target is 21x28
            batch_loss, non_padding = train_batch(model, source, target, optimizer, criterion, attn=attn)
            total_loss += batch_loss * non_padding
            total_observations += non_padding

            if counter % 200 == 0:
                print("batch", str(counter), " : perplexity = ", np.exp((total_loss/total_observations)))
            if stop_after_one_batch:
                return plot_losses
            counter += 1

        print(str(epoch) + "EPOCH LOSS: " + str(total_loss), "PERPLEXITY:", np.exp((total_loss/total_observations)))

        if scheduler:
            scheduler.step()
        plot_losses.append(total_loss)

        print("Validate:", evaluate(model, val_iter, criterion)[0])

        fname = 'seq2seq_3_3_beam_' if filename is None else filename[:-4] 
        torch.save(model.state_dict(), fname + str(epoch) + '.sav')
    return plot_losses

def evaluate(model, val_iter, criterion, attn=False):
    model.eval()
    # model.valid = True
    total_loss = 0.
    total_len = 0.
    for batch in val_iter:
        source, target = process_batch(batch)
        # output, hidden, metadata = model(source, target)
        if attn:
            output, hidden, attention = model(source, target)
        else:
            output, hidden = model(source, target)

        # Take output minus the last character
        output = output[:-1, :, :]

        # Take target without the first character
        target = target[1:, :]

        output_flat = output.view(-1, model.output_size)
        loss = criterion(output_flat, target.view(-1))

        # - 1 hack for nonzero
        non_padding = (target.view(-1) - 1.0).nonzero().size(0)
        # non_padding = (target.view(-1)).ne(1).int().sum()

        # Remove n
        total_loss += non_padding * loss.data
        total_len += non_padding

        if attn:
            visualize(source, output, attention)

    print("Total Loss ", total_loss[0])
    print("Total Len ", total_len)
    print(total_loss[0] / total_len)
    model.train()
    model.valid = False
    return np.exp(total_loss / total_len), output

def kaggle(model, SRC_LANG, TRG_LANG, output_file, input_file='source_test.txt'):
    model.eval()
    model.valid = True
    f = open(input_file, 'r')
    lines = f.readlines()
    with open(output_file, 'w') as out:
        print('id,word', file=out)
        for i, line in enumerate(lines):
            text = Variable(torch.LongTensor([SRC_LANG.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1) # Shape: [len x 1]
            if USE_CUDA:
                text = text.cuda()
            sequences = model(text, None, k=100, use_target=False) # THE ONLY TIME USE_TARGET = FALSE
            # convert each seq to sentence
            print("{}, ".format(i+1), end='', file=out)
            for sequence in sequences:
                sequence = sequence.squeeze()
                english_seq = [TRG_LANG.vocab.itos[j.data[0]] for j in sequence]
                print(english_seq)
                # Only get first 3
                english_seq = english_seq[1:4]
                english_seq = escape("|".join(english_seq))
                print(english_seq, end= ' ',file=out)

            print(file=out)

    model.train()
    model.valid = False

def visualize(sources, outputs, attention):
    for source, output in zip(sources, outputs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c_ax = ax.matshow(attention.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + source.split(' ') + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        plt.close()
