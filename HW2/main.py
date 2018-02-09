import torchtext
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.optim as optim
import trigrams, nnlm
import utils
import random

import torch
CUDA = False

DEBUG = True
train_file = "train.5k.txt" if DEBUG else "train.txt"


# Our input $x$
TEXT = torchtext.data.Field()

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
	path=".", 
	train=train_file, validation="valid.txt", test="valid.txt", text_field=TEXT)

print('len(train)', len(train))

if DEBUG:
	TEXT.build_vocab(train, max_size=2000)
else:
	TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
	(train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)

# coef_1 = 0.0
# # lambdas = [.001, 0, .999]
# values_for_l = [.00001, .0001, .001, .01, .1, .2, .3, .5, .7]
# l_1 = random.choice(values_for_l)
# while l_1 > .1:
# 	l_1 = random.uniform(0,.1)
# l_2 = random.choice(values_for_l)
# while l_1 + l_2 > 1:
# 	l_2 = random.choice(values_for_l)
# l_3 = 1 - l_1 - l_2

# lambdas = [l_1, l_2, l_3] 


# print('lambdas = ', lambdas)
# trigrams_lm = trigrams.TrigramsLM(vocab_size = len(TEXT.vocab), alpha=1, lambdas=lambdas)
# trigrams_lm.train(train_iter, n_iters=None)
# print(utils.validate(trigrams_lm, val_iter))

NNLM = nnlm.LSTMLM(len(TEXT.vocab), 100, 3)
# 
loss_function = nn.NLLLoss()
optimizer = optim.SGD(NNLM.parameters(), lr=0.1)
utils.train(NNLM, train_iter, 1, loss_function, optimizer, hidden=True)
print(utils.validate(NNLM, val_iter, hidden=True))
