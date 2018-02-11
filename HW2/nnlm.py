import torch
import torch.nn as nn
import torch.autograd as autograd

torch.manual_seed(1)

class LSTMLM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n, hidden_size=20):
		super(LSTMLM, self).__init__()
		self.num_layers = 5
		self.hidden_size=hidden_size
		self.embedding = nn.Embedding(vocab_size, embedding_dim) # num_embeddings, embedding_dim
		self.lstm = nn.LSTM(embedding_dim, hidden_size, self.num_layers, batch_first=False)
		self.linear = nn.Linear(hidden_size * n, vocab_size)
		self.init_weights()
		self.initial_hidden = None #autograd.Variable(torch.zeros(1, hidden_size))
		
	def init_weights(self):
		self.embedding.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		
	def forward(self, batch, h):
		# print("input to forward", batch.size()) # [n] 
		# Embed word ids to vectors

		print("batch size",batch.size())
		if torch.cuda.is_available():
			batch = batch.cuda()
			print("batch to cuda")

		print("pre emedding")
		word_vectors = self.embedding(batch) # [n x embedding_dim]
		if torch.cuda.is_available():
			word_vectors.cuda()
		print('hello')
		print("w,h", word_vectors.size(), h[0].size(), h[1].size())
		
		# Get predictions and hidden state from LSTM  
		# print(self.lstm(word_vectors, h)
		out, h = self.lstm(word_vectors, h) # out is [n, 1, hidden_size]
		# print("result of lstm", out.size())
		
		out = out.squeeze().view(-1)
		# print("out.size()", out.size()) # [60]
		# Decode hidden states of all time step
		out = self.linear(out)  
		# print("after linear out.size()", out.size())
		return h, out