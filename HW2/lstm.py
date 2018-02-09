import torch
import torch.nn as nn

class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, layers=1):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.rnn = nn.LSTM(hidden_size, hidden_size, layers)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, inputs, hidden):
		batch_size = inputs.size(0)
		embedding = self.embedding(inputs)
		output, hidden = self.rnn(embedding.view(1, batch_size, -1), hidden)
		output = self.linear(output.view(batch_size, -1))
		return output, hidden