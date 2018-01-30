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
        probs = model(batch.text)
        _, argmax = probs.max(1)
        for i, predicted in enumerate(list(argmax.data)):

            if predicted+1 == batch.label[i].data[0]:
                correct += 1
            total += 1

    return correct / total

class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args

        # Change these names
        vocab_size = args.vocab_size
        embedding_dim = args.embedding_dim
        class_number = args.class_num
        in_channel = 1
        out_channel = args.feature_maps
        filter_windows = args.filter_windows

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Go and understand this
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (F, embedding_dim)) for F in filter_windows])
        self.dropout = nn.Dropout(args.dropout)

        # Go and understand this
        self.fc = nn.Linear(len(filter_windows) * out_channel, class_number)

    # What's going on
    def convolution_max_pool(x, convolution):
        return F.relu(convolution(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):

        embedding = self.embedding(inputs)
        result = [self.convolution_max_pool(embedding, k) for k in self.conv] # k is each filter
        result = self.fc(self.dropout(torch.cat(x, 1))) # Concat and dropout. Why is it called fc?

        return result

parser = argparse.ArgumentParser(description='Text Classification through CNN')
parser.add_argument('-lr', type=float, default=1e-1, help='setting learning rate')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-feature-maps', type=int, default=100, help='number of feature maps')
parser.add_argument('-filter-windows', type=str, default='3,4,5', help='comma-separated filter windows to use for convolution')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
args = parser.parse_args()

if __name__ == '__main__':
    EMBEDDING_SIZE = 10
    LEARN_RATE = 0.1
    FEATURE_WINDOWS = 3, 4, 5
    FEATURE_MAPS = 100
    EMBEDDING_DIM = 128
    DROPOUT = 0.5
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
        (train, val, test), batch_size=50, device=-1, repeat=False)

    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    net = CNN(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(100):
        total_loss = 0
        for batch in train_iter:
            text, label = batch.text, batch.label
            label = label - 1
            net.zero_grad()

            # Figure out if this is the right call
            logit = net(text)
            loss = criterion(log_probs, label)
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
    probs = net(batch.text)
    _, argmax = probs.max(1)
    upload += list(argmax.data)
print("Upload: ", upload)

with open("predictions.txt", "w") as f:
    for u in upload:
        f.write(str(u) + "\n")
