import argparse
import torch
import torchtext
import torch.autograd as autograd
from torchtext.vocab import Vectors, GloVe
import math
import random

import trigrams, nnlm, lstm
import utils
import utilslstm

#toy_text = autograd.Variable(torch.Tensor([[range(1, 11), range(11, 21)], [range(11,21), [22] + list(range(22, 31))]]))

toy_text = autograd.Variable(torch.Tensor([range(1, 11), range(11, 21), range(11,21), [22] + list(range(22, 31))]))

print(toy_text)
trigrams_lm = trigrams.TrigramsLM(vocab_size=100, alpha=1, lambdas=[.1, .4, .5])
# criterion = nn.CrossEntropyLoss()
trigrams_lm.train(toy_text, n_iters=None)
print("UNIGRAMS", trigrams_lm.unigram_counts, "\n")
print("BIGRAMS", trigrams_lm.bigram_counts, "\n",)
print("TRIGRAMS", trigrams_lm.trigram_counts, "\n")


print("UNIGRAM PROBS", trigrams_lm.unigram_probs, "\n")

print("missing ", trigrams_lm.p_ngram(trigrams_lm.unigram_probs, 40, 1))

print("BIGRAM PROBS", trigrams_lm.bigram_probs, "\n")

print("missing 40, 1", trigrams_lm.p_ngram(trigrams_lm.bigram_probs, (40,1), 1))

print("TRIGRAM PROBS", trigrams_lm.trigram_probs, "\n")

print("not missing", trigrams_lm.p_ngram(trigrams_lm.trigram_probs, (11,12,13), 1))

print("missing", trigrams_lm.p_ngram(trigrams_lm.trigram_probs, (21,12,13), 1))

print(toy_text.t())
#print(toy_text.t().contiguous().view(10,4))
print(utils.process_nonbatch(toy_text.t(), 2))