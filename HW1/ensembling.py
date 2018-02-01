import torch.nn as nn
import torch
from torch.autograd import Variable
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import math
import argparse
from math import ceil
import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe
import pdb

binarize_bool = False
torch.manual_seed(42)

# Our input $x$
TEXT = torchtext.data.Field()

# Our labels $y$
LABEL = torchtext.data.Field(sequential=False)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

#for label in range(len(LABEL.vocab)):
#    subset_train = train[]

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=-1, repeat=False)
class CNN(nn.Module):

    def __init__(self, model="non-static", vocab_size=None, embedding_dim=128, class_number=None,
                feature_maps=100, filter_windows=[3,4,5], dropout=0.5):
        super(CNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.class_number = class_number
        self.filter_windows = filter_windows
        self.in_channel = 1
        self.out_channel = feature_maps
        self.model = model

        if model == "static":
            self.embedding.weight.requires_grad = False
        elif model == "multichannel":
            self.embedding2 = nn.Embedding(vocab_size+2, embedding_dim, padding_idx=vocab_size+1)
            self.embedding2.weight.requires_grad = False
            self.in_channel = 2

        self.embedding = nn.Embedding(vocab_size+2, embedding_dim, padding_idx=vocab_size+1)
        self.conv = nn.ModuleList([nn.Conv2d(self.in_channel, self.out_channel, (F, embedding_dim)) for F in filter_windows])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_windows) * self.out_channel, class_number) # Fully connected layer

    def convolution_max_pool(self, inputs, convolution, i, max_sent_len):
        result_convolution = F.relu(convolution(inputs)).squeeze(3) # (batch_size, out_channel, max_seq_len)
        result = F.max_pool1d(result_convolution, result_convolution.size(2)).squeeze(2) # (batch_size, out_channel)
        return result

    def forward(self, inputs):
        # Pad inputs if less than filter window size
        if inputs.size()[1] <= max(self.filter_windows):
            inputs = F.pad(inputs, (1, ceil((max(self.filter_windows)-inputs.size()[1])/2))) # FINISH THIS PADDING
        
        max_sent_len = inputs.size(1)
        embedding = self.embedding(inputs) # (batch_size, max_seq_len, embedding_size)
        embedding = embedding.unsqueeze(1)

        if self.model == "multichannel":
            embedding2 = self.embedding2(inputs)
            embedding2 = embedding2.unsqueeze(1)
            embedding = torch.cat((embedding, embedding2), 1)
        
        result = [self.convolution_max_pool(embedding, k, i, max_sent_len) for i, k in enumerate(self.conv)]
        result = self.fc(self.dropout(torch.cat(result, 1)))
        return result

# Get unigrams from text 
# Get unigram counts (number of each)
# Get sign(Linear)
class NaiveBayes(nn.Module):
    def __init__(self, vocab_size = len(TEXT.vocab), alpha = 1):
        super(NaiveBayes, self).__init__() 
        self.alpha= alpha
        self.p = torch.zeros(vocab_size) + alpha
        self.q = torch.zeros(vocab_size) + alpha
        self.counts = [0, 0]
        self.linear = nn.Linear(vocab_size, 2)
        self.logsoftmax = nn.LogSoftmax() #softmax will turn the output into probabilities, but log is more convnient

    # x is unigrams
    def forward(self, x): #MLP(x) is shorthand for MLP.forward(x)
        x_flatten = x.view(x.size(0), -1) #need to flatten batch_size x 1 x 28 x 28 to batch_size x 28*28
        out = self.hidden_layers(x_flatten)
        out = self.hidden_to_output(out) #you can redefine variables
        return self.logsoftmax(out)
        
nb = NaiveBayes(alpha = 1)

def make_bow_vector(sentence):
    vec = torch.zeros(len(TEXT.vocab))
    for word in sentence:
        vec[word.data] += 1
    return vec.view(1, -1)


def split_classes(b, l):
    if not(b.label[:] == l).any():
        return None
    return (b.text[:, ((b.label[:] == l).nonzero().squeeze())])

# takes in a batch subset as input
def get_feature_counts(batch_subset, p_or_q, binarize=True):
    for row in batch_subset.transpose(0,1):
        seen = set([])
        for x in row:
            if (x.data not in seen) or binarize == False:
                p_or_q[x.data] +=1
                seen.add(x.data)
    return p_or_q

# Takes a single row as input
def feature_count_row(row, binarize=True):
    seen = set([])
    vec = torch.zeros(len(TEXT.vocab))
    for x in row:
        if (x.data not in seen) or binarize == False:
            vec[x.data] +=1
            seen.add(x.data)
    return vec

# TRAIN
for batch in train_iter:
    for label in range(1,3):
        # Select elements in batch with some label
        text_subset = split_classes(batch, label)
        if text_subset is None:
            continue
        # Update number of observations seen with this label
        nb.counts[label - 1] += text_subset.size()[1]
        counts = nb.p if label == 1 else nb.q
        counts = get_feature_counts(text_subset, counts, binarize=binarize_bool)


# Log Conditional class probabilities
# Combine p, q into w: should be VOC x 2
w = torch.stack((nb.p, nb.q), dim=1)
# W needs to be normalized across rows 
# Should still be VOC x2
w = torch.div(w, torch.sum(w, dim=0))
# Take logs
w.log_()


# Log Prior class probabilities
# size is [2]
nb.priors = torch.zeros(2)
nb.priors[0] = nb.counts[0] / float(nb.counts[0] + nb.counts[1])
nb.priors[1] = nb.counts[1] / float(nb.counts[0] + nb.counts[1])
nb.priors.log_()


def validate_nb(val_iter, write=False):
    name = "predictions_nb_alpha=" + str(nb.alpha) + "_bin="+ str(binarize_bool) + ".txt"
    correct, n = 0.0, 0.0
    if write:
        upload = []
        
    # VALIDATION
    for batch in val_iter:
        x = batch.text
        y = batch.label
        # Count vectorize x, reshape to be VOCAB x 10
        # vec = feature_count_row(x).view(-1, 1) 
        batch_features = torch.stack([
            feature_count_row(x_i, binarize=binarize_bool).view(-1) for 
            x_i in torch.unbind(batch.text, dim=1)], dim=1)

        # Pred = w^T x + b for each class
        # w is VOCAB x 2
        # Get Log probabilities [2x 1]
        probs_no_bias = torch.mm(w.t(), batch_features)
        log_probs = probs_no_bias.add(nb.priors.view(2,1))
        # print(log_probs)
        # Get actual prediction 
        nb_vals, nb_preds = torch.max(log_probs, 0)

        # Target is [1,2] but prediction is [0,1]
        target = y - 1

        #print pred.float(), target.data.float()

        cnn_probs = net(batch.text.t_())
        cnn_vals, cnn_preds = cnn_probs.max(1)

        preds = []
        for cnn_val, cnn_pred, nb_val, nb_pred in zip(cnn_vals, cnn_preds, nb_vals, nb_preds):
            if cnn_val.data[0] > nb_val:
                preds.append(cnn_pred.data)
            else:
                preds.append(nb_pred.data)


        for p, t in zip(preds, target):
            if (p[0] == t.data[0]):
                correct += 1
            n +=1

            if write:
                upload.append(p + 1)
    if write:
        with open(name, "w") as f:
            f.write("Id,Cat" + "\n")
            for i, u in enumerate(upload):
                f.write(str(i) + "," + str(u) + "\n")
    print(correct,n)
    return correct/n

def test(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    # test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        _, argmax = probs.max(1)
        upload += list(argmax.data)
        print(upload)

    with open("predictions_ensembling.txt", "w") as f:
        f.write("Id,Cat" + "\n")
        for i, u in enumerate(upload):
            f.write(i, str(u) + "\n")

# Open the CNN 
net = CNN(model='multichannel', vocab_size=len(TEXT.vocab), class_number=2)
net.load_state_dict(torch.load('cnn_lr=0.8'))



print("Validation", validate_nb(val_iter))
print("Test", validate_nb(test_iter, write=False))

#print(1.0 - float(torch.sum(preds - target))/float(preds.size()[1]))
    


