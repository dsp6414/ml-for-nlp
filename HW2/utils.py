import numpy as np
import torchtext 
import torch
import torch.autograd as autograd


def validate(model, val_iter):
	correct = 0.0
	total  = 0.0
	num_zeros = 0.0
	last_pred = None
	for batch in val_iter:
		# Take every row except bottom row = last word in each obs.
		x = batch.text[:-1,:]
		y = batch.text[-1, :].squeeze()
		probs = model(x)
		_, preds = torch.max(probs, 1)

		correct += sum(preds == y.data)
		total += batch.text.size()[1] - 1
		num_zeros += sum(torch.zeros_like(y.data) == y.data)
		print(preds, y)
	print(correct,total, num_zeros)
	return correct / total

def train(model, train_iter, num_epochs, criterion, optimizer, hidden=False):
	if hidden:
		h_0 = autograd.Variable(torch.zeros(model.num_layers * 1, 1, model.hidden_size))
		c_0 = autograd.Variable(torch.zeros(model.num_layers * 1, 1, model.hidden_size))
		h = (h_0, c_0)
	for epoch in range(num_epochs):
		for batch in train_iter:
			# Line vector
			for vector in batch.text.t():
				model.zero_grad()

				x = vector[:-1].data
				y = vector[-1].view(1,1)	

				print("x", x.size())
				
				if hidden:
					h, probs = model.forward(x, h)
				else:
					probs = model.forward(x)
				print("probs", probs)
				print("y", y)
				loss = criterion(probs, y)
				loss.backward()
				optimizer.step()
