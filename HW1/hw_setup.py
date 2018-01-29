# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

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

batch = next(iter(train_iter))
print("Size of text batch [max sent length, batch size]", batch.text.size())
print("Second in batch", batch.text[:, 0])
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 0].data]))
