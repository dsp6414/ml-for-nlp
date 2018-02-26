import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import math
import pdb

import utils

torch.manual_seed(1)

BATCH_SIZE = 128
USE_CUDA = True if torch.cuda.is_available() else False

BOS_EMBED = 2
EOS_EMBED = 3
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.init_param = 0.08

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)

    def init_hidden(self, batch_size=BATCH_SIZE):
        if USE_CUDA:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

    def forward(self, inputs, hidden):
        # seq_len = len(inputs)
        # embedding = self.embedding(inputs).view(seq_len, 1, -1) # check sizes here

        # inputs: [seq_len x batch_sz]
        inverse_inputs = utils.flip(inputs, 0)
        embedding = self.embedding(inverse_inputs) # [len x B x E]
        try:
            output, hidden = self.rnn(embedding, hidden) # [num_layers x batch x hidden]
        except:
            print("INPUTS")
            print(inputs)
            print("EMBEDDING")
            print(embedding)
            print("HIDDEN")
            print(hidden)
            print("RNN")
            print(self.rnn.parameters())
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, last_hidden, encoder_outputs):
        word_embedding = self.embedding(inputs).unsqueeze(0) # [1 x B x N]
        output, hidden = self.rnn(word_embedding, last_hidden)
        # output: [1 x batch x hidden]
        # hidden: [num_layer x batch x hidden], [num_layer x batch x hidden]
        output = output.squeeze(0) # check dim
        output = self.out(output)
        return output, hidden

    def forward_step(self, input_var, last_hidden, encoder_outputs, function=F.log_softmax):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        pdb.set_trace()
        embedded = self.embedding(input_var)
        # embedded = self.dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        predicted_softmax = function(self.out(output.view(-1, self.hidden_size))).view(batch_size, output_size, -1)
        return predicted_softmax, hidden

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        # for i in range(seq_len):
        #     attn_energies[i] = hidden.dot(self.attn(encoder_outputs[i]))

        attn_energies = torch.bmm(hidden.transpose(0, 1), encoder_outputs.transpose(0, 1).transpose(1, 2))

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=2)

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p # need to check if this is a thing
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = Attn(hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(embedding_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # inputs is the true values for the target sentence from previous time step
        # last_hidden is the bottleneck hidden from processing all of encoder
        # encoder_outputs is
    def forward(self, inputs, last_hidden, encoder_outputs):
        word_embedding = self.dropout(self.embedding(inputs)) # [1 x B x Embedding]
        attn_weights = torch.bmm(last_hidden[0].transpose(0, 1), encoder_outputs.transpose(0, 1).transpose(1, 2))
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)
        combined = torch.cat((word_embedding, context), 1)
        #########################
        #  works up until here
        output, hidden = self.rnn(combined.unsqueeze(0), last_hidden)
        output = output.squeeze(0) # B x N (check dimensions)
        output = self.out(torch.cat((output, context), 1))

        return output, hidden, attn_weights

def _inflate(tensor, times, dim):
        """
        Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
        Args:
            tensor: A :class:`Tensor` to inflate
            times: number of repetitions
            dim: axis for inflation (default=0)
        Returns:
            A :class:`Tensor`
        Examples::
            >> a = torch.LongTensor([[1, 2], [3, 4]])
            >> a
            1   2
            3   4
            [torch.LongTensor of size 2x2]
            >> b = ._inflate(a, 2, dim=1)
            >> b
            1   2   1   2
            3   4   3   4
            [torch.LongTensor of size 2x4]
            >> c = _inflate(a, 2, dim=0)
            >> c
            1   2
            3   4
            1   2
            3   4
            [torch.LongTensor of size 4x2]
        """
        repeat_dims = [1] * tensor.dim()
        repeat_dims[dim] = times
        return tensor.repeat(*repeat_dims)

class TopKDecoder(torch.nn.Module):

    def __init__(self, decoder_rnn, k):
        super(TopKDecoder, self).__init__()
        self.rnn = decoder_rnn
        self.k = k
        self.hidden_size = self.rnn.hidden_size
        self.V = self.rnn.output_size
        self.SOS = 2
        self.EOS = 3

    def forward(self, source, target, encoder_outputs, encoder_hidden, use_target=False, function=F.log_softmax,
                    teacher_forcing_ratio=0, retain_output_probs=True):
        """
        Forward rnn for MAX_LENGTH steps.  Look at :func:`seq2seq.models.DecoderRNN.DecoderRNN.forward_rnn` for details.
        """

        inputs = source # Size: 7 x 128
        # Targets: 14 x 128
        max_length = len(target)
        batch_size = len(source[1])


        # encoder_outputs: [source_len x batch x hidden]
        # encoder_hidden: # [num_layers x batch x hidden]
        # decoder_outputs = Variable(torch.zeros(max_length, batch_size, self.output_size))

        decoder_output = Variable(torch.LongTensor([BOS_EMBED] * batch_size))# [1 x batch]
        decoder_hidden = encoder_hidden # [num_layers x batch x hidden]
        self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1)

        # Inflate the initial hidden states to be of size: b*k x h
        # encoder_hidden = self.rnn.init_hidden(batch_size)
        if encoder_hidden is None:
            hidden = None
        else:
            if isinstance(encoder_hidden, tuple):
                hidden = tuple([_inflate(h, self.k, 1) for h in encoder_hidden])
            else:
                hidden = _inflate(encoder_hidden, self.k, 1)

        # ... same idea for encoder_outputs and decoder_outputs
        #if self.rnn.use_attention:
        #    inflated_encoder_outputs = _inflate(encoder_outputs, self.k, 0)
        # else:
        #    inflated_encoder_outputs = None
        inflated_encoder_outputs = None

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = Variable(sequence_scores)

        if USE_CUDA:
            sequence_scores = sequence_scores.cuda()

        # Initialize the input vector
        input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1))

        if USE_CUDA:
            input_var = input_var.cuda()

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()
        pdb.set_trace()

        for _ in range(0, max_length):
            # Run the RNN one step forward
            pdb.set_trace()
            log_softmax_output, hidden = self.rnn.forward_step(input_var, hidden, 
                inflated_encoder_outputs, function=function)

            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_softmax_output)

            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = _inflate(sequence_scores, self.V, 1)
            sequence_scores += log_softmax_output.squeeze(1)
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(batch_size * self.k, 1)
            sequence_scores = scores.view(batch_size * self.k, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)
            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
            stored_hidden.append(hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_size)

        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in output]
        if isinstance(h_n, tuple):
            decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
        else:
            decoder_hidden = h_n[:, :, 0, :]
        metadata = {}
        metadata['inputs'] = inputs
        metadata['output'] = output
        metadata['h_t'] = h_t
        metadata['score'] = s
        metadata['topk_length'] = l
        metadata['topk_sequence'] = p
        metadata['length'] = [seq_len[0] for seq_len in l]
        metadata['sequence'] = [seq[0] for seq in p]
        return decoder_outputs, decoder_hidden, metadata

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.
        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state
        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates
            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = nw_hidden[0][0].size()
            h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
        l = [[self.rnn.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
                                                                # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b   # the number of EOS found
                                    # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.k)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    if lstm:
                        current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]
                        current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]
                        h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data
                        h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data
                    else:
                        current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                        h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.data[0]] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for h in step]) for step in reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in reversed(h_t)]
            h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
        s = s.data

        return output, h_t, h_n, s, l, p

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
            score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, n_layers=1, dropout=0.0, attn=False, beam=False, k=5):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.init_param = 0.08
        self.attn = attn
        self.encoder = EncoderRNN(input_size, embedding_size, hidden_size, n_layers, dropout)

        if attn:
            self.decoder = AttnDecoderRNN(embedding_size, hidden_size, output_size, n_layers, dropout)
        else:
            self.decoder = DecoderRNN(embedding_size, hidden_size, output_size, n_layers, dropout)

        self.beam = beam
        if self.beam:
            self.beam_decoder = TopKDecoder(self.decoder, k)
        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            if self.beam:
                self.beam_decoder.cuda()

        self.valid = False

    def forward(self, source, target, use_target=False):
        max_length = len(target)
        batch_size = len(source[1])

        encoder_hidden = self.encoder.init_hidden(batch_size=batch_size) # can insert batch size here
        encoder_outputs, encoder_hidden = self.encoder(source, encoder_hidden)
        # encoder_outputs: [source_len x batch x hidden]
        # encoder_hidden: # [num_layers x batch x hidden]
        decoder_outputs = Variable(torch.zeros(max_length, batch_size, self.output_size))
        if USE_CUDA:
            decoder_outputs = decoder_outputs.cuda()

        # decoder_output = Variable(target[0].data) # [1 x batch]
        decoder_output = Variable(torch.LongTensor([BOS_EMBED] * batch_size))# [1 x batch]
        decoder_hidden = encoder_hidden # [num_layers x batch x hidden]
        # decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        if USE_CUDA:
            decoder_output = decoder_output.cuda()
            # decoder_context = decoder_context.cuda()

        if self.beam and self.valid:
            decoder_outputs, decoder_hidden, metadata = self.beam_decoder(source, target, encoder_outputs, encoder_hidden, use_target=False, function=F.log_softmax,
                    teacher_forcing_ratio=0, retain_output_probs=True)
            pdb.set_trace()
            return decoder_outputs, decoder_hidden


        # Greedy search
        for i in range(0, max_length):
            if self.attn:
                decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_output, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_outputs)
            # decoder_output: [batch x len(EN)]
            # target: [target_len x batch]
            decoder_outputs[i] = decoder_output
            if use_target:
                decoder_output = target[i].cuda() if USE_CUDA else target[i]
            else:
                decoder_output = decoder_output.max(1)[1]
        # decoder_output, hidden = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden # decoder_output [target_len x batch x en_vocab_sz]


