import torchtext
from torchtext.vocab import Vectors

import trigrams
import utils

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
	TEXT.build_vocab(train, max_size=1000)
else:
	TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
	(train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)

trigrams_lm = trigrams.TrigramsLM(vocab_size = len(TEXT.vocab), alpha=1)
trigrams_lm.train(train_iter, debug=True)
print(utils.validate(trigrams_lm, val_iter))
