
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

print(val[:5])