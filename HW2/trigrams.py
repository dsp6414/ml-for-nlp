class TrigramLM(nn.Module):
    def __init__(self, lambdas, vocab_size):
        super(TrigramLM, self).__init__()
        
        self.lambdas = lambdas
        self.vocab_size = vocab_size
        self.unigram_probs = torch.zeros(self.vocab_size) # vocab_size
        self.bigram_probs = torch.zeros(self.vocab_size, self.vocab_size)
        self.trigram_probs = torch.zeros(self.vocab_size, self.vocab_size, self.vocab_size)
        
    def forward(self, input, hidden):
        unigram = self.input[-1]
        return output, hidden