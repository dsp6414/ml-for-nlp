import torch
import torch.nn as nn
import torch.autograd as autograd

torch.manual_seed(1)

DROPOUT = 0.5

class LSTMLM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n, hidden_size=60):
		super(LSTMLM, self).__init__()
		self.n = n
		self.num_layers = 1
		self.hidden_size = hidden_size
		self.embedding_dim = embedding_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.tanh = nn.Tanh()
		self.lstm = nn.LSTM(embedding_dim, hidden_size, self.num_layers, batch_first=False)
		self.dropout = nn.Dropout(DROPOUT)
		self.linear = nn.Linear(hidden_size * n, vocab_size)
		self.linear_two = nn.Linear(self.embedding_dim * n, vocab_size)
		self.inner_lin = nn.Linear(self.embedding_dim * n, self.embedding_dim * n)
		self.init_weights()
		self.initial_hidden = None
		self.is_eval = False

	def eval(self):
		self.is_eval = True
		
	def init_weights(self):
		self.linear.bias.data.fill_(0)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear_two.bias.data.fill_(0)
		self.linear_two.weight.data.uniform_(-0.1, 0.1)
		
	def forward(self, batch):
		# Embed word ids to vectors
		word_vectors = (self.embedding(batch)) # [n x embedding_dim]

		if not self.is_eval:
			word_vectors = self.dropout(word_vectors)

		if word_vectors.dim() == 1:	
			word_vectors = word_vectors.unsqueeze(1)
		if torch.cuda.is_available():
			word_vectors = word_vectors.cuda()
		
		# Get predictions and hidden state from LSTM  
		out = word_vectors.t().contiguous().view(1, -1, self.n * self.embedding_dim)
		out = self.tanh(out)
		out = self.inner_lin(out)
		out = self.tanh(out)
		out = self.linear_two(out)
		out = self.dropout(out)	
		return out.squeeze()