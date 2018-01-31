import argparse
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
        self.embedding = nn.Embedding(vocab_size+2, embedding_dim, padding_idx=vocab_size+1)

        # Go and understand this
        self.conv = nn.ModuleList([nn.Conv2d(self.in_channel, self.out_channel, (F, embedding_dim), stride=embedding_dim) for F in filter_windows])
        # self.conv = nn.ModuleList([nn.Conv1d(self.in_channel, self.out_channel, embedding_dim * F, stride=embedding_dim) for F in filter_windows])
        self.dropout = nn.Dropout(dropout)

        # Go and understand this
        self.fc = nn.Linear(len(filter_windows) * self.out_channel, class_number) # Fully connected layer

        if model == "static":
            self.embedding.weight.requires_grad = False
        elif model == "multichannel":
            self.embedding2 = nn.Embedding(vocab_size+2, embedding_dim, padding_idx=vocab_size+1)
            self.embedding2.weight.requires_grad = False
            in_channel = 2

    def convolution_max_pool(self, inputs, convolution, i, max_sent_len):
        # result_convolution = F.relu(convolution(inputs))
        # result = F.max_pool1d(result_convolution, max_sent_len - self.filter_windows[i] + 1).view(-1, self.out_channel)
        
        # OLD CODE
        # pdb.set_trace()

        result_convolution = F.relu(convolution(inputs)).squeeze(3) # (batch_size, out_channel, max_seq_len)
        
        # pdb.set_trace()
        result = F.max_pool1d(result_convolution, result_convolution.size(2)).squeeze(2) # (batch_size, out_channel)
        return result

    def forward(self, inputs):
        max_sent_len = inputs.size(1)

        embedding = self.embedding(inputs) # (batch_size, max_seq_len, embedding_size)
        if self.model == "multichannel":
            embedding2 = self.embedding2(inputs)
            embedding = torch.cat((embedding, embedding2), 1)

        embedding = embedding.unsqueeze(1) # () might be a problem in multichannel

        # embedding = self.embedding(inputs).view(-1, 1, self.embedding_dim * inputs.size(1))

        # if self.model == "multichannel":
            # embedding2 = self.embedding2(inputs).view(-1, 1, self.embedding_dim * inputs.size(1))
            # embedding = torch.cat((embedding, embedding2), 1)

        # pdb.set_trace()
        result = [self.convolution_max_pool(embedding, k, i, max_sent_len) for i, k in enumerate(self.conv)]
        
        # pdb.set_trace()
        result = self.fc(self.dropout(torch.cat(result, 1)))

        return result

        # OLD CODE

        # x = [F.relu(conv(embedding)).squeeze(3) for conv in self.conv]
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # Passing in max_sentence_length - filter_window + 1
        # result = self.fc(self.dropout(torch.cat(x, 1)))
        # return result 

parser = argparse.ArgumentParser(description='Text Classification through CNN')
parser.add_argument('-lr', type=float, default=1e-1, help='setting learning rate')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-feature-maps', type=int, default=100, help='number of feature maps')
parser.add_argument('-filter-windows', type=str, default='3,4,5', help='comma-separated filter windows to use for convolution')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
args = parser.parse_args()

if __name__ == '__main__':
    EMBEDDING_SIZE = 128
    MAX_NORM = 3

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
        (train, val, test), batch_size=25, device=-1, repeat=False)

    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    net = CNN(vocab_size=len(TEXT.vocab), class_number=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(1):
        total_loss = 0
        for batch in train_iter:
            text, label = batch.text.t_(), batch.label
            # if len(text) < max(self.filter_windows):
            #     text = F.pad(text, (1, ceiling((max(self.filter_windows)-len(text))/2)), value=) # FINISH THIS PADDING
            label = label - 1
            net.zero_grad()

            # pdb.set_trace()
            # Figure out if this is the right call
            logit = net(text)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        print("loss =", total_loss)

    for param in net.parameters():
        print(param)

    print(validate(net, val_iter))
    
# TESTING
"All models should be able to be run with following command."
upload = []
# Update: for kaggle the bucket iterator needs to have batch_size 10
# test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, repeat=False)
for batch in test_iter:
    # Your prediction data here (don't cheat!)
    probs = net(batch.text.t_())
    _, argmax = probs.max(1)
    upload += list(argmax.data)
print("Upload: ", upload)

with open("predictions.txt", "w") as f:
    for u in upload:
        f.write(str(u) + "\n")
