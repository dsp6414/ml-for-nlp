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

def validate(model, val_iter):
	correct, n = 0.0, 0.0
	for batch in val_iter:
		for x,y in zip(batch.text,  batch.label):
			bow_vec = autograd.Variable(make_bow_vector(x))
			target = y - 1
			log_probs = logreg((bow_vec))
			_, predicted = torch.max(log_probs.data, 1)
			if torch.equal((target.float()), Variable(predicted.float())):
				correct += 1
			n +=1
	return correct/n

print("Training")

for i in range(100):
	#print(i)
	for batch in train_iter:
		batch_bows = torch.stack([
			make_bow_vector(x_i).view(-1) for 
			x_i in torch.unbind(batch.text, dim=1)], dim=1)
		print(batch_bows.size())
		#print(batch_bows, batch_bows.size())
		for x,y in zip(batch.text.t(),  batch.label):
			#print(x,y)
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			logreg.zero_grad()

			# Step 2. Make our BOW vector and also we must wrap the target in a
			# Variable as an integer. For example, if the target is SPANISH, then
			# we wrap the integer 0. The loss function then knows that the 0th
			# element of the log probabilities is the log probability
			# corresponding to SPANISH
			print(x.size())
			bow_vec = autograd.Variable(make_bow_vector(x))

			#print(bow_vec)
			target = y - 1

			# Step 3. Run our forward pass.
			log_probs = logreg(bow_vec)

			#print(log_probs)

			# Step 4. Compute the loss, gradients, and update the parameters by
			# calling optimizer.step()
			loss = loss_function(log_probs, target)
			#print(loss)
			loss.backward()
			optimizer.step()
		#print(validate(logreg))

print("Done training")

# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable


print("Validation accuracy", validate(logreg, val_iter))
print("Test accuracy", validate(logreg, test_iter))

