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

torch.manual_seed(42)

def validate(model, val_iter):
    correct, total = 0.0, 0.0
    for batch in val_iter:
        probs = model(batch.text.t_())
        _, argmax = probs.max(1)
        for i, predicted in enumerate(list(argmax.data)):
            if predicted+1 == batch.label[i].data[0]:
                correct += 1
            total += 1
    return correct / total

class CNN(nn.Module):

    def __init__(self, model="non-static", vocab_size=None, embedding_dim=256, class_number=None,
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
            self.embedding2 = nn.Embedding(vocab_size+2, embedding_dim)
            self.embedding2.weight.requires_grad = False
            self.in_channel = 2

        self.embedding = nn.Embedding(vocab_size+2, embedding_dim)
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
        embedding = embedding.unsqueeze(1) # (batch_size, 1, max_seq_len, embedding_size)

        if self.model == "multichannel":
            embedding2 = self.embedding2(inputs)
            embedding2 = embedding2.unsqueeze(1)
            embedding = torch.cat((embedding, embedding2), 1)
        
        result = [self.convolution_max_pool(embedding, k, i, max_sent_len) for i, k in enumerate(self.conv)]
        result = self.fc(self.dropout(torch.cat(result, 1)))
        return result

if __name__ == '__main__':
    # Our input $x$
    TEXT = torchtext.data.Field()
    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False)

    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=50, device=-1, repeat=False)

    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))


    net = CNN(model='multichannel', vocab_size=len(TEXT.vocab), class_number=2)
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adadelta(parameters, lr=0.5)


    for epoch in range(50):
        total_loss = 0
        for batch in train_iter:
            text, label = batch.text.t_(), batch.label
            label = label - 1
            net.zero_grad()

            logit = net(text)
            loss = criterion(logit, label)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=3)
            optimizer.step()

            total_loss += loss.data

    print("VAL SET", validate(net, val_iter))


# TEST SET
"All models should be able to be run with following command."
upload = []
# Update: for kaggle the bucket iterator needs to have batch_size 10
# test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, repeat=False)
correct, total = 0.0, 0.0
for batch in test_iter:
    # Your prediction data here (don't cheat!)
    probs = net(batch.text.t_())
    _, argmax = probs.max(1)
    for i, predicted in enumerate(list(argmax.data)):
        if predicted+1 == batch.label[i].data[0]:
            correct += 1
        total += 1

    upload += list(argmax.data)
print("TEST SET:", correct / total)
# print("Upload: ", upload)

with open("predictions.txt", "w") as f:
    for u in upload:
        f.write(str(u) + "\n")
