import argparse
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe
import random

import trigrams, nnlm, lstm
import utils
import utilslstm

import pdb

BATCH_SIZE = 20
BPTT = 35
EMBEDDING_SIZE = 128
NUM_LAYERS = 2
# LR = 1 * 1.2 # decreased by 1.2 for each epoch after 6th
# DECAY = 1.2
# TEMP_EPOCH = 6
# EPOCHS = 39

# Large LSTM
EPOCHS = 55
LR = 1 * 1.15 # decreased by 1.15 for each epoch after 14th
DECAY = 1.15
TEMP_EPOCH = 14
GRAD_NORM = 10

parser = argparse.ArgumentParser(description='Language Modeling')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of RNN')
parser.add_argument('--mini', type=bool, default=False, help='run smaller dataset')
args = parser.parse_args()

CUDA = False
if torch.cuda.is_available():
	CUDA = True

DEBUG = True if args.mini else False
train_file = "train.5k.txt" if DEBUG else "train.txt"

# Our input $x$
TEXT = torchtext.data.Field()
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
	(train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=BPTT, repeat=False)

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

if args.model == 'NNLM':
	NNLM = nnlm.LSTMLM(len(TEXT.vocab), 100, 3)
	criterion = nn.NLLLoss()
	optimizer = optim.SGD(NNLM.parameters(), lr=0.1)
	utils.train(NNLM, train_iter, 1, criterion, optimizer, hidden=True)
	print(utils.validate(NNLM, val_iter, hidden=True))

if args.model == 'LSTM':
	rnn = lstm.LSTM(embedding_size=EMBEDDING_SIZE, vocab_size=len(TEXT.vocab), num_layers=NUM_LAYERS, lstm_type='large')
	if torch.cuda.is_available():
		print("USING CUDA")
		rnn = rnn.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adadelta(rnn.parameters(), lr=LR/DECAY)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[TEMP_EPOCH], gamma=1/DECAY)
	print("TRAINING DATA")
	utilslstm.train(rnn, train_iter, EPOCHS, criterion, optimizer, scheduler=scheduler, grad_norm=10) #change grad norm

	print("SAVING MODEL")
	filename = 'lstm_large.sav'
	torch.save(rnn.state_dict(), filename)

	print("LOADING MODEL")
	loaded_model = lstm.LSTM(embedding_size=EMBEDDING_SIZE, vocab_size=len(TEXT.vocab), num_layers=NUM_LAYERS, lstm_type='large')
	loaded_model.load_state_dict(torch.load(filename))
	print("VALIDATION SET")
	utilslstm.evaluate(loaded_model, val_iter, criterion)
	# print("TEST SET")
	# utilslstm.evaluate(loaded_model, test_iter, criterion)

