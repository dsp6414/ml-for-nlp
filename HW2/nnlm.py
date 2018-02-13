import torch
import torch.nn as nn
import torch.autograd as autograd

torch.manual_seed(1)

class LSTMLM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n, hidden_size=60):
		super(LSTMLM, self).__init__()
		self.n = n
		self.num_layers = 1
		self.hidden_size=hidden_size
		self.embedding = nn.Embedding(vocab_size, embedding_dim) # num_embeddings, embedding_dim
		self.tanh = nn.Tanh()
		self.lstm = nn.LSTM(embedding_dim, hidden_size, self.num_layers, batch_first=False)
		self.dropout = nn.Dropout(0.5)
		self.linear = nn.Linear(hidden_size * n, vocab_size)
		self.init_weights()
		self.initial_hidden = None #autograd.Variable(torch.zeros(1, hidden_size))
		self.is_eval = False

	def eval(self):
		self.is_eval = True
		
	def init_weights(self):
		self.embedding.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		
	def forward(self, batch, h):
		# print("input to forward", batch.size()) # [n] 
		# Embed word ids to vectors

		word_vectors = (self.embedding(batch)) # [n x embedding_dim]

		if not self.is_eval:
			word_vectors = self.dropout(word_vectors)

		if word_vectors.dim() == 1:	
			word_vectors = word_vectors.unsqueeze(1)
		if torch.cuda.is_available():
			word_vectors = word_vectors.cuda()
		
		# Get predictions and hidden state from LSTM  
		# print(self.lstm(word_vectors, h)
		print(word_vectors.t().size()) # Word vectors is: [640 x 3 x 100] -> [3 x 640 x 100]
		out, h = self.lstm(word_vectors.t(), h) # out is [n, 1, hidden_size]
		out = self.dropout(out)	
		#print(out.size())
		#out = out.view(-1, self.hidden_size * self.n)
		# print("out.size()", out.size()) # [60]
		# Decode hidden states of all time step
		#out = self.linear(out)  
		# print("after linear out.size()", out.size())
		return h, out.squeeze()