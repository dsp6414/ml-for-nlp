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
    output_flat = output.view(-1, model.output_size) # [(tg_len x batch) x en_vocab_sz]
    # not sure whether to use ground truth target or network's prediction
    loss = criterion(output_flat, target.view(-1))
    loss.backward()

    # figure out how to do this
    # if L2 Norm of Gradient / 128 > 5, then g = 5g/s
    nn.utils.clip_grad_norm(model.parameters(), CLIP)
    optimizer.step()
    return loss.data[0]

def train(model, train_iter, epochs, optimizer, criterion, scheduler=None, filename=None): # do I need a max_length=MAX_LENGTH?
    model.train()
    plot_losses = []
    counter = 0

    stop_after_one_batch = False
    if epochs == 0:
        epochs = 1
        stop_after_one_batch = True


    for epoch in range(epochs):
        total_loss = 0
        for batch in train_iter:
            source, target = process_batch(batch) # Source is 11x28, target is 21x28
            batch_loss = train_batch(model, source, target, optimizer, criterion)
            total_loss += batch_loss

            if counter % 50 == 0:
                print(str(counter) + " counter: " + str(total_loss))
            if stop_after_one_batch:
                return plot_losses
            counter += 1

        print(str(epoch) + "EPOCH LOSS: " + str(total_loss))

        if scheduler:
            scheduler.step()
        plot_losses.append(total_loss)

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
        output, hidden, metadata = model(source, target)
        output_flat = output.view(-1, model.output_size)
        loss = criterion(output_flat, target.view(-1))
        total_loss += len(source) * loss.data
        total_len += len(source)

    print("Total Loss ", total_loss[0])
    print("Total Len ", total_len)
    print(total_loss[0] / total_len)
    model.train()
    model.valid = False
    print(metadata)
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

def kaggle(model, output_file, input_file='source_test.txt'):
    pdb.set_trace()
    model.eval()
    model.valid = True
    f = open(input_file)
    lines = f.readlines()
    with open(outputfile, 'w') as out:
        print('id,word', file=out)
        for i, line in enumerate(lines):
            text = Variable(torch.LongTensor([TEXT.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1)
            if CUDA:
                text = text.cuda()
            fake_target = Variable(torch.LongTensor([0] * 20))
            output, hidden, metadata = model(source, target)
            pdb.set_trace()
            print("%d,%s"%(i+1, " ".join([TEXT.vocab.itos[i.data[0]] for i in indices[:20]])), file=out)

    model.train()
    model.valid = False

