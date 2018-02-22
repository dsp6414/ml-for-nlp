from torchtext import data
from torchtext import datasets
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
import utils, model

torch.manual_seed(1)

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 20
BATCH_SIZE = 128
TEMP_EPOCH = 5
EPOCHS = 7.5

N_LAYERS = 4
HIDDEN = 1000
EMBEDDING = 1000
LR = 0.7


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
print(DE.vocab.freqs.most_common(10))
print("Size of German vocab", len(DE.vocab)) # 13353
print(EN.vocab.freqs.most_common(10))
print("Size of English vocab", len(EN.vocab)) # 11560
print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) #vocab index for <s> (2), </s> (3)

train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
												  repeat=False, sort_key=lambda x: len(x.src))

# batch = next(iter(train_iter))
# print("Source")
# print(batch.src)
# print("Target")
# print(batch.trg)

# Fix these!!
#encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
#decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

optimizer = optim.SGD(encoder.parameters(), lr=LR)
milestones = list(range(TEMP_EPOCH, EPOCHS - 1, 0.5))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1/DECAY)

