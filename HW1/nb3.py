import torch.nn as nn
import torch
from torch.autograd import Variable
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import math

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

# Get unigrams from text 
# Get unigram counts (number of each)
# Get sign(Linear)
class NaiveBayes(nn.Module):
    def __init__(self, vocab_size = len(TEXT.vocab), alpha = 1):
        super(NaiveBayes, self).__init__() 
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
        
nb = NaiveBayes()

def make_bow_vector(sentence):
    vec = torch.zeros(len(TEXT.vocab))
    for word in sentence:
        vec[word.data] += 1
    return vec.view(1, -1)

def one_hot(batch,depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,batch)

def split_classes(b, l):
    if not(b.label[:] == l).any():
        return None
    return (b.text[:, ((b.label[:] == l).nonzero().squeeze())])

# takes in a batch subset as input
def get_feature_counts(batch_subset, p_or_q):
    for row in batch_subset.transpose(0,1):
        seen = set([])
        for x in row:
            if x.data not in seen:
                p_or_q[x.data] +=1
                seen.add(x.data)
    return p_or_q

# Takes a single row as input
def feature_count_row(row):
    seen = set([])
    vec = torch.zeros(len(TEXT.vocab))
    for x in row:
        if x.data not in seen:
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
        counts = get_feature_counts(text_subset, counts)

print(nb.p, nb.q)

# Log Conditional class probabilities
# Combine p, q into w: should be VOC x 2
w = torch.stack((nb.p, nb.q), dim=1)
# W needs to be normalized across rows 
# Should still be VOC x2
w = torch.div(w, torch.sum(w, dim=0))
# Take logs
w.log_()
print(w, w.size())

# Log Prior class probabilities
# size is [2]
nb.priors = torch.zeros(2)
nb.priors[0] = nb.counts[0] / float(nb.counts[0] + nb.counts[1])
nb.priors[1] = nb.counts[1] / float(nb.counts[0] + nb.counts[1])
nb.priors.log_()
print(nb.priors, nb.priors.size())

def validate_nb(val_iter):
    correct, n = 0.0, 0.0
    # VALIDATION
    for batch in val_iter:
        for x,y in zip(batch.text, batch.label):  
            # Count vectorize x, reshape to be VOCAB x1
            vec = feature_count_row(x).view(-1, 1) 
            # Pred = w^T x + b for each class
            # Get Log probabilities [2x 1]
            log_probs = torch.mm(w.t(), vec).view(-1) + nb.priors
            # print(log_probs)
            # Get actual prediction 
            _, pred = torch.max(log_probs, 0)

            # Target is [1,2] but prediction is [0,1]
            target = y - 1

            #print pred.float(), target.data.float()

            if torch.equal(pred.float(), target.data.float()):
                correct += 1
            n +=1
    print(correct,n)
    return correct/n

print("Validation", validate_nb(val_iter))
print("Test", validate_nb(test_iter))

#print(1.0 - float(torch.sum(preds - target))/float(preds.size()[1]))
    


