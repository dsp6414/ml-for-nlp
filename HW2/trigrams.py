import torch
import torch.nn as nn

class TrigramsLM(nn.Module):
	def __init__(self, vocab_size, lambdas = [1./100.,39./100.,6./10.], alpha=0):
		super(TrigramsLM, self).__init__()
		
		self.alphas = lambdas
		self.alpha = alpha
		self.vocab_size = vocab_size
		self.unigram_counts = {}
		self.bigram_counts = {}
		self.trigram_counts = {}
		self.unigram_probs = {}
		self.bigram_probs = {}
		self.trigram_probs = {}

	def p_ngram(self, ngram_dict, ngram, n):
		if ngram in ngram_dict:
			# Ignore unigrams with really high counts
			return ngram_dict[ngram]
		else:
			if self.alpha == 0:
				return 0
			if n == 1:
				denom = self.vocab_size * self.alpha + self.sum_unigrams
			elif n == 2:
				prev_unigram, i = ngram
				denom = self.vocab_size * self.alpha + self.unigram_counts[prev_unigram]
			elif n == 3:
				b1, b2, i = ngram
				if (b1, b2) not in self.bigram_counts:
					denom = self.alpha * self.vocab_size
				else:
					denom = self.vocab_size * self.alpha + self.bigram_counts[(b1, b2)]	
			return self.alpha/ denom
	def forward(self, input_data):

		# Batch shape is bptt, batch_size
		# Assume input has observations in rows, transpose it to be observations in columns
		input_data = input_data.t() 
		last_unigrams = input_data[-1, :] # size = batch_size
		last_bigrams = input_data[-2:, :] # size = 2 x batch_size
		batch_size = last_unigrams.size()[0]
		# print(batch_size)
		# print(batch_size)
		# print(last_unigrams, last_bigrams)
		# Unigram probabilities

		def p_i(i, prev_unigram, prev_bigram):
			b1 = prev_bigram[0]
			b2 = prev_bigram[1]
			return (self.alphas[0] * self.p_ngram(self.unigram_probs, i, 1) + 
			self.alphas[1] * self.p_ngram(self.bigram_probs, (prev_unigram,i), 2) +
			self.alphas[2] * self.p_ngram(self.trigram_probs, (b1, b2, i), 3))

		preds = []
		for i in range(batch_size):
			unigram = last_unigrams[i].data[0]
			bigram = last_bigrams[:, i].data # 2 x 1
			# print(unigram, bigram)
			# p_unigrams = self.unigram_probs # vocab_size
			# p_bigrams = self.bigram_probs[unigram] # vocab_size
			# p_trigrams = self.trigram_probs[bigram[0], bigram[1]] # vocab_size
			#pred_j  =
			#pred = self.alphas[0] * p_unigrams + self.alphas[1] * p_bigrams + self.alphas[2] * p_trigrams
			pred = [p_i(j, unigram, bigram) for j in range(self.vocab_size)] 
			# preds.append(pred.squeeze())
			preds.append(torch.Tensor(pred))

		tensor_preds = torch.stack(preds)
		return tensor_preds

	def set_lambdas(self, lambdas):
		self.alphas = lambdas

	# I know Sasha said not to put training in the model, but how else would it work for trigrams??
	def train(self, train_iter, n_iters=None):
		def update_dict(d, val):
			if val in d:
				d[val] += 1
			else:
				d[val] = 1
		# Assume batches to be batch_size, max_bptt
		batch_num = 0
		for batch in train_iter:
			if n_iters is not None and batch_num > n_iters:
				break
			x = batch.text
			# x = batch
			# Update unigram counts
			for row in x:
				for word in row:
					update_dict(self.unigram_counts, word.data[0])

			# Update bigram counts
			for row in x:
				for j in range(len(row) - 1):
					w_t_2 = row[j].data[0]
					w_t_1 = row[j + 1].data[0]
					update_dict(self.bigram_counts, (w_t_2, w_t_1))
					# self.bigram_probs[w_t_2, w_t_1] += 1

			# Update trigram counts
			for row in x:
				for j in range(len(row) - 2):
					w_t_3 = row[j].data[0]
					w_t_2 = row[j + 1].data[0]
					w_t_1 = row[j + 2].data[0] # Most recently seen word
					update_dict(self.trigram_counts, (w_t_3, w_t_2, w_t_1))
					# self.trigram_probs[w_t_3, w_t_2, w_t_1] += 1
			batch_num += 1

		# Transform counts into probabilities
		# Normalize trigram counts using bigram sums
		for trigram, count in self.trigram_counts.items():
			w_t_3, w_t_2, w_t_1 = trigram
			self.trigram_probs[trigram] = (count + self.alpha) / float(self.bigram_counts[(w_t_3, w_t_2)] + self.vocab_size * self.alpha)

		# Normalize bigram counts with unigram sums
		for bigram, count in self.bigram_counts.items():
			w_t_2, w_t_1 = bigram
			# self.trigram_probs['missing'] = alpha / float(count + self.vocab_size * alpha)
			self.bigram_probs[bigram] = (count + self.alpha) / (float(self.unigram_counts[w_t_1]) + self.vocab_size * self.alpha)

		# Normalize unigram counts with laplace smoothing
		self.sum_unigrams = sum(self.unigram_counts.values())
		for unigram, count in self.unigram_counts.items():
			self.unigram_probs[unigram] = (count + self.alpha) / (self.sum_unigrams + float(self.vocab_size * self.alpha))
		print("done training")

		