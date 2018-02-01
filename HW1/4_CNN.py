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

    def __init__(self, model="non-static", vocab_size=None, embedding_dim=128, class_number=None,
                feature_maps=100, filter_windows=[3,4,5], dropout=0.5):
        super(CNN, self).__init__()

        # Change these names
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
        # self.conv = nn.ModuleList([nn.Conv1d(self.in_channel, self.out_channel, embedding_dim * F, stride=embedding_dim) for F in filter_windows])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_windows) * self.out_channel, class_number) # Fully connected layer

    def convolution_max_pool(self, inputs, convolution, i, max_sent_len):
        ############ OLD CODE
        # result_convolution = F.relu(convolution(inputs))
        # result = F.max_pool1d(result_convolution, result_convolution.size(2)).squeeze(2)
        # return result
        ############

        # CODE THAT WORKS
        ########
        # pdb.set_trace()
        result_convolution = F.relu(convolution(inputs)).squeeze(3) # (batch_size, out_channel, max_seq_len)
        
        # pdb.set_trace()
        result = F.max_pool1d(result_convolution, result_convolution.size(2)).squeeze(2) # (batch_size, out_channel)
        return result
        ##########

    def forward(self, inputs):

        # CODE THAT WORKS
        ########
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
        ##########

        # OLD CODE
        ###################
        # max_sent_len = inputs.size(1)
        # embedding = self.embedding(inputs).view(-1, 1, self.embedding_dim * max_sent_len)
        # if self.model == "multichannel":
        #     embedding2 = self.embedding2(inputs).view(-1, 1, self.embedding_dim * inputs.size(1))
        #     embedding = torch.cat((embedding, embedding2), 1)

        # result = [self.convolution_max_pool(embedding, k, i, max_sent_len) for i, k in enumerate(self.conv)]
        # result = self.fc(self.dropout(torch.cat(result, 1)))
        # return result
        ###################

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

    learning_rate = [0.01, 0.1, 0.5, 0.8, 1]
    # saved_nets = []

    for lr in learning_rate:
        net = CNN(model='multichannel', vocab_size=len(TEXT.vocab), class_number=2)
        criterion = nn.CrossEntropyLoss()
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.Adadelta(parameters, lr=lr)
        # Tune epochs thorugh early stopping (test on the validation set until the percentage goes down)
        for epoch in range(50):
            counter = 0
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
            print("loss =", total_loss)

        print("LR VAL SET", validate(net, val_iter))
        filename = 'cnn_lr=' + str(lr)
        torch.save(net.state_dict(), filename)


# TESTING
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
