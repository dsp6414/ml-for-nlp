import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe

binarize_bool = True
dec = .25
lrate = .5
mom = 0

hyperparams = {'bin': binarize_bool, 'reg': dec, 'lr': lrate, 'momentum':mom}
torch.manual_seed(42)

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
print("GRADIENT DESCENT = Adadelta")
optimizer = torch.optim.Adadelta(logreg.parameters(), lr = lrate) #initialize with parameters
loss_function = nn.NLLLoss()
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
	(train, val, test), batch_size=10, device=-1, repeat=False)

def make_bow_vector(sentence, binarize=True):
	seen = set([])
	vec = torch.zeros(len(TEXT.vocab))
	for word in sentence:
		if (word not in seen) or (binarize==False):
			seen.add(word.data)
			vec[word.data] += 1
	return vec.view(1, -1)


def validate(model, val_iter):
	correct, n = 0.0, 0.0
	for batch in val_iter:
		for x,y in zip(batch.text.t(),  batch.label):
			bow_vec = autograd.Variable(make_bow_vector(x))
			target = y - 1
			log_probs = logreg((bow_vec))
			_, predicted = torch.max(log_probs.data, 1)
			if torch.equal((target.float()), Variable(predicted.float())):
				correct += 1
			n +=1
	return correct/n

print("HYPERPARAMETERS: ", hyperparams)
print("Training")

for i in range(100):
	for batch in train_iter:
		batch_bows = torch.stack([
			make_bow_vector(x_i).view(-1) for 
			x_i in torch.unbind(batch.text, dim=1)], dim=1).t()

		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		logreg.zero_grad()

		# Step 2. Make our BOW vector and also we must wrap the target in a
		# Variable as an integer. For example, if the target is SPANISH, then
		# we wrap the integer 0. The loss function then knows that the 0th
		# element of the log probabilities is the log probability
		# corresponding to SPANISH
		#print(batch_bows.size())
		# bow_vec = autograd.Variable(make_bow_vector(x))
		bow_vecs = autograd.Variable(batch_bows)

		#print(bow_vec)
		target = batch.label - 1

		#print(target)

		# Step 3. Run our forward pass.
		log_probs = logreg(bow_vecs)

		#print(log_probs)

		# Step 4. Compute the loss, gradients, and update the parameters by
		# calling optimizer.step()
		loss = loss_function(log_probs, target)
		#print(loss)
		loss.backward()
		optimizer.step()
		#print(validate(logreg))
	if i in [1, 2, 3, 5, 7, 9, 10, 15, 20, 30, 50]:
		print("EPOCH:", i, "; validation accuracy = ", validate(logreg, val_iter), "; test accuracy = ", validate(logreg, test_iter))


print("Done training")

# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable


print("Validation accuracy", validate(logreg, val_iter))
print("Test accuracy", validate(logreg, test_iter))

