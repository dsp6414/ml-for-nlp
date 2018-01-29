
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe

import pbd

torch.manual_seed(42)

def make_context_vector(text):
    tensor = torch.LongTensor(context)
    return autograd.Variable(tensor), target

# make_context_vector(data[0][0], word_to_ix)  # example

class CBOW(nn.Module):

    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).sum(dim=0) #Unsure why I need to do this
        out = self.linear1(embeddings)
        log_probs = F.log_softmax(out)
        return log_probs

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

    net = CBOW(CONTEXT_SIZE, embedding_size=EMBEDDING_SIZE, vocab_size=len(TEXT.vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    # Training for CBOW

    train_iter1, val_iter1, test_iter1 = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=10, device=-1)

    data = []
    for batch in train_iter1:
        text = batch.text
        for i in range(2, len(text)-2):
            context = [text[i-2], text[i-1],
                        text[i+1], text[i+2]]
            target = text[i]
            data.append((context, target))

    pbd.trace()

    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            net.zero_grad()
            context_var = make_context_vector(context)
            log_probs = net(context_var)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        print(total_loss)

    # # Training for sentiment classification
    # for epoch in range(50):
    #     total_loss = 0
    #     for batch in train_iter:
    #         text, label = batch.text, batch.label
    #         net.zero_grad()

    #         # Currently a batch, not an individual sentence
    #         context_var, target = make_context_vector(text)
    #         log_probs = net(context_var)

    #         loss = criterion(log_probs, target)

    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.data
    #     print("loss =", total_loss)
