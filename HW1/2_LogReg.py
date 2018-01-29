import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe

class LogReg(nn.Module):

	def __init__(self, vocab_size):
		super(LogReg, self).__init__()
		self.linear = nn.Linear(vocab_size, 2)
		
	def forward(self, inputs):
		out = self.linear(inputs)
		log_probs = F.logsigmoid(out)
		return log_probs

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

logreg = LogReg(len(TEXT.vocab))
optimizer = torch.optim.SGD(logreg.parameters(), lr = 0.1) #initialize with parameters
loss_function = nn.NLLLoss()
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
	(train, val, test), batch_size=10, device=-1, repeat=False)

def make_bow_vector(sentence):
	vec = torch.zeros(len(TEXT.vocab))
	for word in sentence:
		vec[word.data] += 1
	return vec.view(1, -1)

def one_hot(batch,depth):
	ones = torch.sparse.torch.eye(depth)
	return ones.index_select(0,batch)

print("Training")
for batch in train_iter:
	for x,y in zip(batch.text,  batch.label):
		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		logreg.zero_grad()

		# Step 2. Make our BOW vector and also we must wrap the target in a
		# Variable as an integer. For example, if the target is SPANISH, then
		# we wrap the integer 0. The loss function then knows that the 0th
		# element of the log probabilities is the log probability
		# corresponding to SPANISH
		bow_vec = autograd.Variable(make_bow_vector(x))
		target = y - 1

		# Step 3. Run our forward pass.
		log_probs = logreg(bow_vec)

		# Step 4. Compute the loss, gradients, and update the parameters by
		# calling optimizer.step()
		loss = loss_function(log_probs, target)
		loss.backward()
		optimizer.step()

print("Done training")
correct, n = 0.0, 0.0
# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
for batch in val_iter:
	for x,y in zip(batch.text,  batch.label):
		bow_vec = autograd.Variable(make_bow_vector(x))
		target = y - 1
		log_probs = logreg((bow_vec))
		_, predicted = torch.max(log_probs.data, 1)
		#print(log_probs)
		#print(predicted)
		#print(target, predicted)
		if torch.equal((target.float()), Variable(predicted.float())):
			correct += 1
		n +=1

print("Validation accuracy", correct/n, correct, n)
