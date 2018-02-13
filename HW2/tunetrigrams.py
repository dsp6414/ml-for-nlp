import torchtext
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.optim as optim
import trigrams, nnlm
import utils
import random

import torch
CUDA = False

DEBUG = False
train_file = "train.5k.txt" if DEBUG else "train.txt"


# Our input $x$
TEXT = torchtext.data.Field()

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
	path=".", 
	train=train_file, validation="valid.txt", test="valid.txt", text_field=TEXT)

if DEBUG:
	TEXT.build_vocab(train, max_size=2000)
else:
	TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
	(train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)

trigrams_lm = trigrams.TrigramsLM(vocab_size = len(TEXT.vocab), alpha=1)
trigrams_lm.train(train_iter, n_iters=None)
criterion = nn.CrossEntropyLoss()
while(1):
	l_1 = random.uniform(0,.5)
	l_2 = random.uniform(0,1)
	while l_1 + l_2 > 1:
		l_1 = random.uniform(0,1)
		l_2 = random.uniform(0,1)
	l_3 = 1 - l_1 - l_2

	lambdas = [l_1, l_2, l_3] 
	trigrams_lm.set_lambdas(lambdas)

	print('lambdas = ', lambdas)
	print("VALIDATION SCORE", utils.validate_trigrams(trigrams_lm, val_iter, criterion, max_iters = 10))