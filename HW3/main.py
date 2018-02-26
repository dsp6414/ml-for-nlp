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
import os.path
import pdb

torch.manual_seed(1)

BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 20
BATCH_SIZE = 128
TEMP_EPOCH = 5
# EPOCHS = 7.5
EPOCHS = 5

N_LAYERS = 1
# N_LAYERS = 4
HIDDEN = 1000
EMBEDDING = 1000
LR = 0.7

USE_CUDA = True if torch.cuda.is_available() else False

DE = data.Field(tokenize=utils.tokenize_de)
EN = data.Field(tokenize=utils.tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS
p
arser = argparse.ArgumentParser(description='Translation')
parser.add_argument('--beam', type=bool, default=False, help='use beam search')
parser.add_argument('--attn', type=bool, default=False, help='use attention')
parser.add_argument('--modelpath', type=str, default=None, help='load a model')
args = parser.parse_args()


# Try to save the files
# train_file = 'train.sav'
# val_file = 'val.sav'

# if not os.path.exists(train_file):
#     train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
#                                          filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
#                                          len(vars(x)['trg']) <= MAX_LEN)
#     MIN_FREQ = 5
#     DE.build_vocab(train.src, min_freq=MIN_FREQ)
#     EN.build_vocab(train.trg, min_freq=MIN_FREQ)

#     try:
#         torch.save(train, train_file)
#         torch.save(val, val_file)
#     except Exception as e:
#         print(e)
# else:
#     train = torch.load(train_file)
#     val = torch.load(val_file)

# print(DE.vocab.freqs.most_common(10))
# print("Size of German vocab", len(DE.vocab)) # 13353
# print(EN.vocab.freqs.most_common(10))
# print("Size of English vocab", len(EN.vocab)) # 11560
# print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) #vocab index for <s> (2), </s> (3)

train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)

print("Finish build vocab")

train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                  repeat=False, sort_key=lambda x: len(x.src))

# batch = next(iter(train_iter))
# print("Source")
# print(batch.src)
# print("Target")
# print(batch.trg)
print("Done bucketing data")

# Fix these!!
model = Seq2Seq(len(DE.vocab), len(EN.vocab), EMBEDDING, HIDDEN, N_LAYERS, attn=args.attn, beam=args.beam)
if USE_CUDA:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
# milestones = list(range(TEMP_EPOCH, EPOCHS - 1, 0.5))
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1/DECAY)

filename = args.model_path if args.model_path else 'seq2seq_2_25_.sav'
if os.path.exists(filename):
    model.load_state_dict(torch.load(filename))
else:
    plot_losses = utils.train(model, train_iter, EPOCHS, optimizer, criterion)
    print(plot_losses)
    torch.save(model.state_dict(), filename)

print("EVALUATE")
loss, output = utils.evaluate(model, val_iter, criterion)
for row in output.data:
    print(" ".join([EN.vocab.itos[i] for i in row.max(1)[1]])) # check this later

print("KAGGLE")
utils.kaggle(model, output_file="seq2seq_preds.txt", input_file="source_test.txt")
