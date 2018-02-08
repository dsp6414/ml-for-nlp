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

it = iter(train_iter)
batch = next(it) 
#print("Size of text batch [max bptt length, batch size]", batch.text.size()) #(32L, 10L) = max tensor length, batchsize
#print("Second in batch", batch.text[:, 2])
#print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))

print("bigrams", batch.text[-3:, :])

trigrams_lm = trigrams.TrigramsLM(vocab_size = len(TEXT.vocab), alpha=1)
trigrams_lm.train(train_iter, debug=DEBUG)
print(utils.validate(trigrams_lm, val_iter))
