
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe

import pdb

torch.manual_seed(42)

def test(model, test_iter):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    # test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, repeat=False)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        _, argmax = probs.max(1)
        upload += list(argmax.data)
    print("Upload: ", upload)

    with open("predictions.txt", "w") as f:
        for u in upload:
            f.write(str(u) + "\n")

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

class CBOW(nn.Module):

    def __init__(self, embedding_size=100, vocab_size=None, num_labels=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, num_labels)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).sum(dim=0) #Unsure why I need to do this
        out = self.linear1(embeddings)
        log_probs = F.log_softmax(out, dim=0)

        return log_probs

if __name__ == '__main__':
    EMBEDDING_SIZE = 128

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
        (train, val, test), batch_size=10, device=-1, repeat=False)

    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    net = CBOW(embedding_size=EMBEDDING_SIZE, vocab_size=len(TEXT.vocab), num_labels=2)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    # Training for sentiment classification
    losses = []
    for epoch in range(100):
        total_loss = torch.Tensor([0])
        for batch in train_iter:
            text, label = batch.text, batch.label
            label = label - 1
            net.zero_grad()

            # Currently a batch, not an individual sentence
            log_probs = net(text)
            loss = criterion(log_probs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        losses.append(total_loss)
    print(losses)


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