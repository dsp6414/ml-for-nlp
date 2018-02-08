import torch
import torch.nn as nn

class TrigramsLM(nn.Module):
	def __init__(self, vocab_size, lambdas = [1./100.,39./100.,6./10.], alpha=0):
		super(TrigramsLM, self).__init__()
		

		self.alphas = lambdas
		self.vocab_size = vocab_size
		self.unigram_probs = torch.zeros(self.vocab_size) + alpha # vocab_size
		self.bigram_probs = torch.zeros(self.vocab_size, self.vocab_size) + alpha
		self.trigram_probs = torch.zeros(self.vocab_size, self.vocab_size, self.vocab_size) + alpha

		
	def forward(self, input_data):
		# Batch shape is bptt, batch_size
		# Assume input has observations in columns 
		last_unigrams = input_data[-1, :] # size = batch_size
		last_bigrams = input_data[-2:, :] # size = 2 x batch_size

		batch_size = last_unigrams.size()[0]
		# Unigram probabilities

		preds = []
		for i in range(batch_size):
			unigram = last_unigrams[i].data
			bigram = last_bigrams[:, i].data # 2 x 1
			p_unigrams = self.unigram_probs # vocab_size
			p_bigrams = self.bigram_probs[unigram] # vocab_size
			p_trigrams = self.trigram_probs[bigram[0], bigram[1]] # vocab_size
			pred = self.alphas[0] * p_unigrams + self.alphas[1] * p_bigrams + self.alphas[2] * p_trigrams
			preds.append(pred.squeeze())

		tensor_preds = torch.stack(preds)
		# print(tensor_preds)
		return tensor_preds
	
	# I know Sasha said not to put training in the model, but how else would it work for trigrams??
	def train(self, train_iter, n_iters=None):
		# Transposed batches to be batch_size, max_bptt
		batch_num = 0
		for batch in train_iter:
			if n_iters is not None and batch_num > n_iters:
				break
			x = batch.text.t() 
			# Update unigram counts
			for row in x:
				for word in row:
					self.unigram_probs[word.data] += 1

			# Update bigram counts
			for row in x:
				for j in range(len(row) - 1):
					w_t_2 = row[j].data
					w_t_1 = row[j + 1].data
					self.bigram_probs[w_t_2, w_t_1] += 1

			# Update trigram counts
			for row in x:
				for j in range(len(row) - 2):
					w_t_3 = row[j].data
					w_t_2 = row[j + 1].data
					w_t_1 = row[j + 2].data # Most recently seen word
					self.trigram_probs[w_t_3, w_t_2, w_t_1] += 1
			batch_num += 1

		# Transform counts into probabilities
		self.unigram_probs = self.unigram_probs / float(torch.sum(self.unigram_probs))
		
		bigram_counts = torch.sum(self.bigram_probs, dim=1, keepdim=True).float()
		self.bigram_probs = self.bigram_probs / bigram_counts

		trigram_counts = torch.sum(self.trigram_probs, dim=2).float()
		self.trigram_probs = self.trigram_probs/ trigram_counts

		print("unigram max", max(self.unigram_probs))
		print("bigram max", torch.max(self.bigram_probs))
		print('trigram max', torch.max(self.trigram_probs))


	