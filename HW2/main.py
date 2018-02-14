import argparse
import math
import random
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe

import trigrams, nnlm, lstm
import utils, utilslstm

torch.manual_seed(287)

BATCH_SIZE = 20
BPTT = 35
EMBEDDING_SIZE = 128
NUM_LAYERS = 2

# Medium LSTM
# LR = 1 * 1.2 # decreased by 1.2 for each epoch after 6th
# DECAY = 1.2
# TEMP_EPOCH = 6
# EPOCHS = 39
# GRAD_NORM. = 5

# Large LSTM
LR = 1 * 1.15 # decreased by 1.15 for each epoch after 14th
DECAY = 1.15
TEMP_EPOCH = 14
EPOCHS = 55
GRAD_NORM = 10

parser = argparse.ArgumentParser(description='Language Modeling')
parser.add_argument('--model', type=str, default='LSTM',
					help='type of RNN')
parser.add_argument('--mini', type=bool, default=False, help='run smaller dataset')
parser.add_argument('--path', type=str, default=None, help='load a past model')
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

if DEBUG:
	TEXT.build_vocab(train, max_size=2000)
else:
	TEXT.build_vocab(train)

if args.model == 'extension':
	BATCH_SIZE = 40

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
	(train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=BPTT, repeat=False)

def kaggle(model, file, outputfile=None):
	f = open(file)
	lines = f.readlines()
	hidden = model.init_hidden()
	if outputfile is None:
		outputfile = 'sample.txt'
	with open(outputfile, 'w') as out:
		print('id,word', file=out)
		for i, line in enumerate(lines):
			text = Variable(torch.LongTensor([TEXT.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1)
			if CUDA:
				text = text.cuda()
			h = model.init_hidden(batch_size=1)
			probs, h = model(text, h) # probs: [10 x vocab_size]
			values, indices = torch.sort(probs[-1], descending=True)
			print("%d,%s"%(i+1, " ".join([TEXT.vocab.itos[i.data[0]] for i in indices[:20]])), file=out)

def kaggle_trigrams(model, file, output):
	f = open(file)
	lines = f.readlines()
	with open(output, 'w') as out:
		print('id,word', file=out)
		for i, line in enumerate(lines):
			text = Variable(torch.LongTensor([TEXT.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1)
			if CUDA:
				text = text.cuda()
			probs = Variable(model(text.t())) # probs: [10 x vocab_size]
			values, indices = torch.sort(probs[-1], descending=True)
			print("%d,%s"%(i+1, " ".join([TEXT.vocab.itos[i.data[0]] for i in indices[:20]])), file=out)

def kaggle_nnlmgrams(model, file, output):
	f = open(file)
	lines = f.readlines()
	with open(output, 'w') as out:
		print('id,word', file=out)
		for i, line in enumerate(lines):
			text = Variable(torch.LongTensor([TEXT.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1)
			if CUDA:
				text = text.cuda()
			probs = Variable(model(text)) # probs: [10 x vocab_size]
			values, indices = torch.sort(probs[-1], descending=True)
			print("%d,%s"%(i+1, " ".join([TEXT.vocab.itos[i.data[0]] for i in indices[:20]])), file=out)

def ensembled_kaggle(model_lstm, model_trigrams, file):
	f = open(file)
	lines = f.readlines()
	with open('ensembeld.txt', 'w') as out:
		print('id,word', file=out)
		for i, line in enumerate(lines):
			text = Variable(torch.LongTensor([TEXT.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1)
			print("text", text, text.size)
			if CUDA:
				text = text.cuda()
			probs = Variable(model(text.t())) # probs: [10 x vocab_size]
			values, indices = torch.sort(probs[-1], descending=True)
			print("%d,%s"%(i+1, " ".join([TEXT.vocab.itos[i.data[0]] for i in indices[:20]])), file=out)
				
if args.model == 'NNLM':
	if args.path is not None:
		NNLM = nnlm.LSTMLM(len(TEXT.vocab), 60, 3)
		if CUDA:
			print("converting NNLM to cuda")
			NNLM = NNLM.cuda()
		NNLM.load_state_dict(torch.load('nnlm_two_layers_ten_iter_sixtyembed_with_vectors.sav'))

		kaggle_nnlmgrams(NNLM, 'input.txt', 'NNLM_preds.txt')
	else:
		url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
		TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
		NNLM = nnlm.LSTMLM(len(TEXT.vocab), 60, 3)
		if CUDA:
			print("converting NNLM to cuda")
			NNLM.cuda()

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(NNLM.parameters(), lr=0.3)
		utils.train(NNLM, train_iter, 10, criterion, optimizer, hidden=False)

		# Saving Model
		filename = 'nnlm_two_layers_ten_iter_sixtyembed_with_vectors.sav'
		torch.save(NNLM.state_dict(), filename)
		kaggle(NNLM, 'input.txt', 'NNLM_preds.txt')
		print("perplex", utils.validate(NNLM, val_iter, criterion, hidden=True))

elif args.model == 'Trigrams':
	trigrams_lm = trigrams.TrigramsLM(vocab_size = len(TEXT.vocab), alpha=0.01, lambdas=[.2, .5, .3])
	criterion = nn.CrossEntropyLoss()
	trigrams_lm.train(train_iter, n_iters=None)
	print(utils.validate_trigrams(trigrams_lm, val_iter, criterion))
	print("Calculate Kaggle")
	kaggle_trigrams(trigrams_lm, "input.txt", "trigramsagain.txt")

elif args.model == 'Ensemble':
	print("TRAINING TRIGRAMS MODEL")
	trigrams_lm = trigrams.TrigramsLM(vocab_size = len(TEXT.vocab), alpha=1, lambdas=[.1, .4, .5])
	criterion = nn.CrossEntropyLoss()
	trigrams_lm.train(train_iter, n_iters=None)

	filename = 'lstm_large_hidden45.sav'
	print("LOADING LSTM MODEL")
	loaded_model = lstm.LSTM(embedding_size=EMBEDDING_SIZE, vocab_size=len(TEXT.vocab), num_layers=NUM_LAYERS, lstm_type='large')
	if CUDA:
		print("USING CUDA")
		loaded_model = loaded_model.cuda()
	loaded_model.load_state_dict(torch.load(filename))
	criterion = nn.CrossEntropyLoss()
	print("VALIDATION SET")
	loss = utilslstm.evaluate2(loaded_model, val_iter, criterion)

elif args.model == 'LSTM':
	# Save Model
	# rnn = lstm.LSTM(embedding_size=EMBEDDING_SIZE, vocab_size=len(TEXT.vocab), num_layers=NUM_LAYERS, lstm_type='large')
	# if CUDA:
	# 	print("USING CUDA")
	# 	rnn = rnn.cuda()
	# criterion = nn.CrossEntropyLoss()
	# optimizer = optim.Adadelta(rnn.parameters(), lr=LR/DECAY)
	# milestones = list(range(TEMP_EPOCH, EPOCHS - 1))
	# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1/DECAY)
	
	# # Train data
	# utilslstm.train(rnn, train_iter, EPOCHS, criterion, optimizer, scheduler=scheduler, grad_norm=10) #change grad norm

	# # Save model
	# filename = 'lstm_large_hidden.sav'
	# torch.save(rnn.state_dict(), filename)

	# Load Model
	filename = 'lstm_large_hidden.sav'
	print("LOADING MODEL")
	loaded_model = lstm.LSTM(embedding_size=EMBEDDING_SIZE, vocab_size=len(TEXT.vocab), num_layers=NUM_LAYERS, lstm_type='large')
	if CUDA:
		print("USING CUDA")
		loaded_model = loaded_model.cuda()
	loaded_model.load_state_dict(torch.load(filename))
	criterion = nn.CrossEntropyLoss()
	print("VALIDATION SET")
	# loss = utilslstm.evaluate(loaded_model, val_iter, criterion)
	# print("Perplexity")
	# print(math.exp(loss))
	print("Calculate Kaggle")
	kaggle(loaded_model, 'input.txt')

elif args.model == 'extension':
	rnn = lstm.LSTMExtension(embedding_size=400, vocab_size=len(TEXT.vocab), num_layers=2)
	if CUDA:
		print("USING CUDA")
		rnn = rnn.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adadelta(rnn.parameters(), lr=0.5)

	# Train data
	utilslstm.train(rnn, train_iter, 50, criterion, optimizer, grad_norm=0.25) #change grad norm

	# Save Model
	filename = 'lstm_extension_new.sav'
	torch.save(rnn.state_dict(), filename)

	# Load Model
	filename = 'lstm_extension_new.sav'
	print("LOADING MODEL")
	loaded_model = lstm.LSTMExtension(embedding_size=400, vocab_size=len(TEXT.vocab), num_layers=2)
	loaded_model.load_state_dict(torch.load(filename))
	if CUDA:
		print("USING CUDA")
		loaded_model = loaded_model.cuda()
	criterion = nn.CrossEntropyLoss()
	print("VALIDATION SET")
	loss = utilslstm.evaluate(loaded_model, val_iter, criterion)
	print("Perplexity")
	print(math.exp(loss))
	print("Calculate Kaggle")
	kaggle(loaded_model, 'input.txt')
