from torchtext import data
from torchtext import datasets
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
import utils
from model import Seq2Seq

torch.manual_seed(1)

BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 20
BATCH_SIZE = 128
TEMP_EPOCH = 5
# EPOCHS = 7.5
EPOCHS = 1

N_LAYERS = 4
HIDDEN = 1000
EMBEDDING = 1000
LR = 0.7

USE_CUDA = True if torch.cuda.is_available() else False

DE = data.Field(tokenize=utils.tokenize_de)
EN = data.Field(tokenize=utils.tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
										 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
										 len(vars(x)['trg']) <= MAX_LEN)
print(train.fields)
print(len(train))
print(vars(train[0]))

MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)
# print(DE.vocab.freqs.most_common(10))
# print("Size of German vocab", len(DE.vocab)) # 13353
# print(EN.vocab.freqs.most_common(10))
# print("Size of English vocab", len(EN.vocab)) # 11560
# print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) #vocab index for <s> (2), </s> (3)

train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
												  repeat=False, sort_key=lambda x: len(x.src))

# batch = next(iter(train_iter))
# print("Source")
# print(batch.src)
# print("Target")
# print(batch.trg)
print("Done bucketing data")

# Fix these!!
model = Seq2Seq(len(DE.vocab), len(EN.vocab), EMBEDDING, HIDDEN, N_LAYERS)
if USE_CUDA:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
# milestones = list(range(TEMP_EPOCH, EPOCHS - 1, 0.5))
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1/DECAY)

plot_losses = utils.train(model, train_iter, EPOCHS, optimizer, criterion)
print(plot_losses)
