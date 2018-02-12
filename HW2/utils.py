import numpy as np
import torchtext 
import torch
import torch.autograd as autograd
import torch.nn as nn

torch.manual_seed(1)
NNLM_GRAD_NORM = 5


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

def validate(model, val_iter, criterion, hidden=False):
	correct = 0.0
	total  = 0.0
	num_zeros = 0.0
	n_vectors = 0
	for batch in val_iter:
		processed_batch = process_batch(batch, 3)
		if torch.cuda.is_available():
			processed_batch = processed_batch.cuda()
		print(processed_batch)
		if hidden:
			h_0 = (torch.zeros(model.num_layers * 1, processed_batch.size()[0], model.hidden_size))
			c_0 = (torch.zeros(model.num_layers * 1, processed_batch.size()[0], model.hidden_size))
			h = (h_0, c_0)
		if torch.cuda.is_available():
			h = (h_0.cuda(), c_0.cuda())


		x = processed_batch[:, :-1]
		y = processed_batch[:, -1]

		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
		if hidden:
			h, probs = model(x, h)
		else:
			probs = model(x)
			# Probs is 1-d if you go vector by vector
		_, preds = torch.max(probs, 1)

		loss = criterion(probs, y)
		loss_total += loss
		# total += batch.text.size()[1] - 1
		total += y.size()[0]
		num_zeros += sum(torch.zeros_like(y) == y)
		# print(preds, y)

	mean_loss = loss /float(total)
	return( 2.0 ** mean_loss)

def train(model, train_iter, num_epochs, criterion, optimizer, scheduler=None, hidden=False):
	print("TRAINING")
	for epoch in range(num_epochs):
		loss_total = 0.0
		n_iters = 0
		n_obs = 0.0
		for batch in train_iter:
			print(n_iters)
			processed_batch = autograd.Variable(process_batch(batch, 3))
			if hidden:
				if torch.cuda.is_available():
					h_0 = autograd.Variable(torch.zeros(model.num_layers * 1, processed_batch.size()[0], model.hidden_size)).cuda()
					c_0 = autograd.Variable(torch.zeros(model.num_layers * 1, processed_batch.size()[0], model.hidden_size)).cuda()
					h = (h_0, c_0)
				else:
					h_0 = autograd.Variable(torch.zeros(model.num_layers * 1, processed_batch.size()[0], model.hidden_size))
					c_0 = autograd.Variable(torch.zeros(model.num_layers * 1, processed_batch.size()[0], model.hidden_size))
					h = (h_0, c_0)
			if torch.cuda.is_available():
				processed_batch = processed_batch.cuda()
			# about 200 rows and 4 columns
			model.zero_grad()

			x = processed_batch[:, :-1] # 200 x3 
			y = processed_batch[:, -1] # 200 x 1

			if hidden:
				h, probs = model.forward(x, h)
			else:
				probs = model.forward(x)

			# probs = probs.view(1, -1)
			loss = criterion(probs, y)
			loss_total += loss.data[0]
			n_obs += processed_batch.size()[0]
			loss.backward()
			nn.utils.clip_grad_norm(model.parameters(), max_norm=NNLM_GRAD_NORM)
			optimizer.step()
			n_iters +=1

		# take avg of losses
		# loss_avg = loss_total / float(n_obs)
		# print("perplexity", 2.0 ** loss_avg)

	print("done training")
