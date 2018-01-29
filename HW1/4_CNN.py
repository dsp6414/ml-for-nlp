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
parser = argparse.ArgumentParser(description='Text Classification through CNN')

parser.add_argument('-lr', type=float, default=1e-1, help='setting learning rate')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
args = parser.parse_args()

class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args

        # Change these names
        vocab_size = args.vocab_size
        embedding_dim = args.embedding_dim
        class_number = args.class_num
        in_channel = 1
        out_channel = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Go and understand this
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)

        # Go and understand this
        self.fc = nn.Linear(len(kernel_sizes) * out_channel, class_number)

    # What's going on
    def convolution_max_pool(x, convolution):
        return F.relu(convolution(x).permute(0, 2, 1).max(1)[0])

    def forward(self, input):

        embed = self.embedding(input)
        result = [self.convolution_max_pool(embed, k) for k in self.conv]
        result = self.fc(self.dropout(torch.cat(x, 1))) # Concat and dropout

        return result

if __name__ == '__main__':
    CONTEXT_SIZE = 4
    EMBEDDING_SIZE = 10

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
        (train, val, test), batch_size=10, device=-1)

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
            net.zero_grad()

            # Figure out if this is the right call
            logit = net(text)

            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        print(total_loss)

