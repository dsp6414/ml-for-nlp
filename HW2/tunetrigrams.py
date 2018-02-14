import torchtext
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.optim as optim
import trigrams, nnlm
import utils
import random
from torch.autograd import Variable as Variable

import torch
CUDA = False

DEBUG = False
train_file = "train.5k.txt" if DEBUG else "train.txt"

def kaggle_trigrams(model, file, output):
	f = open(file)
	lines = f.readlines()
	with open(output, 'w') as out:
		print('id,word', file=out)
		for i, line in enumerate(lines):
			text = Variable(torch.LongTensor([TEXT.vocab.stoi[word] for word in line.split(' ')[:-1]])).unsqueeze(1)
			# print("text", text, text.size) # [10 x 1] -> text.t is [1 x 10]
			if torch.cuda.is_available():
				text = text.cuda()
			probs = Variable(model(text.t())) # probs: [10 x vocab_size]
			values, indices = torch.sort(probs[-1], descending=True)
			print("%d,%s"%(i+1, " ".join([TEXT.vocab.itos[i.data[0]] for i in indices[:20]])), file=out)
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


bad_unigrams = [TEXT.vocab.stoi[word] for word in ['<unk>', '<eos>']]

print(bad_unigrams)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
	(train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)

trigrams_lm = trigrams.TrigramsLM(vocab_size = len(TEXT.vocab), alpha=1)
trigrams_lm.train(train_iter, n_iters=None)
criterion = nn.CrossEntropyLoss()

while True:
	l_1 = random.uniform(0,.01)
	l_2 = random.uniform(0,1)
	while l_1 + l_2 > .99:
		l_1 = random.uniform(0,1)
		l_2 = random.uniform(0,1)
	l_3 = 1 - l_1 - l_2

	lambdas = [l_1, l_2, l_3] 
	lambdas = [.2, .5, .3]
	trigrams_lm.set_lambdas(lambdas)

	print('lambdas = ', lambdas)
	str_to_append = str(round(l_1,5)) + '_' + str(round(l_2,5)) + '_' + str(round(l_3,5))
	str_to_append = 'elbert'
	file_name = "trigrams_capped_" + str_to_append + ".txt"
	kaggle_trigrams(trigrams_lm, "input.txt", file_name)
	print("KAGGLE TRIGRAMS")
	# print("VALIDATION SCORE", utils.validate_trigrams(trigrams_lm, val_iter, criterion, max_iters = 10))