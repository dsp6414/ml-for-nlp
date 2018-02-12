import numpy as np
import torchtext 
import torch
import torch.autograd as autograd

torch.manual_seed(1)

# Batch will have the observations vertically.
def process_batch(batch, n):
	n_rows = batch.text.size()[1]

	xs = []
	ys = []
	# Iterate through the batch observations
	for row_num, row in enumerate(batch.text.t()):
		# Get words n at a time
		for i in range(len(row) - n):
			# get ith through i +2nd words (inclusive)
			x = row[i: i+n].data
			y = row[i+n].data
			xs.append(x)
			ys.append(y)
	xs = torch.stack([x_i for x_i in xs])
	ys = torch.stack([y_i for y_i in ys])

	# Return batch horizontally (each row is obs, last column is label)
	return(torch.cat((xs, ys), dim=1))

def validate(model, val_iter, hidden=False):
	correct = 0.0
	total  = 0.0
	num_zeros = 0.0
	if hidden:
		h_0 = autograd.Variable(torch.zeros(model.num_layers * 1, 1, model.hidden_size))
		c_0 = autograd.Variable(torch.zeros(model.num_layers * 1, 1, model.hidden_size))
		h = (h_0, c_0)
		if torch.cuda.is_available():
			h = (h_0.cuda(), c_0.cuda())

	n_vectors = 0
	for batch in val_iter:
		processed_batch = process_batch(batch, 3)
		if torch.cuda.is_available():
			processed_batch = processed_batch.cuda()
		print(processed_batch)
		# print("processed shape", processed_batch.size(), processed_batch)
		for vector in autograd.Variable(processed_batch):
			if n_vectors > 5000:
				print("correct", correct)
				print("total", total)
				return(correct / total)
			x = vector[:-1]
			y = vector[-1].view(1)
			if hidden:
				h, probs = model(x, h)
			else:
				probs = model(x)
				# Probs is 1-d if you go vector by vector
			_, preds = torch.max(probs, 0)

			print("probs",probs)
			print((preds==y)[0])
			correct += torch.sum(preds.data == y.data)
			# total += batch.text.size()[1] - 1
			total += 1
			num_zeros += sum(torch.zeros_like(y.data) == y.data)
			n_vectors += 1
		# print(preds, y)
	print(correct,total, num_zeros)
	return correct / total

def train(model, train_iter, num_epochs, criterion, optimizer, scheduler=None, hidden=False):
	print("TRAINING")
	for epoch in range(num_epochs):
		if hidden:
			h_0 = autograd.Variable(torch.zeros(model.num_layers * 1, 1, model.hidden_size))
			c_0 = autograd.Variable(torch.zeros(model.num_layers * 1, 1, model.hidden_size))
			h = (h_0, c_0)
			if torch.cuda.is_available():
				h = (h_0.cuda(), c_0.cuda())

		n_iters = 0
		for batch in train_iter:
			# Line vector
			processed_batch = autograd.Variable(process_batch(batch, 3))
			for vector in processed_batch:
				print(n_iters)
				if n_iters > 10:
					print("good enough")
					return
				model.zero_grad()

				x = vector[:-1]
				y = vector[-1].view(1)

				if torch.cuda.is_available():
					x = x.cuda()
					y = y.cuda()

				if hidden:
					h, probs = model.forward(x, h)
				else:
					probs = model.forward(x)

				probs = probs.view(1, -1)
				loss = criterion(probs, y)
				loss.backward(retain_graph=True)
				optimizer.step()
				n_iters +=1

	print("done training")
