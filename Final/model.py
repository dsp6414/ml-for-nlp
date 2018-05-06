import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from corpus import WORD_INDEX
import logging

N_PROP_TYPES = 8
N_PROP_OBJECTS = 35

SOS = 1
EOS = 2

MAX_LEN = 20

def print_tensor(data):
    for x in data:
        logging.info([WORD_INDEX.get(word) for word in x])


# 1-d tensor
def pad_end1d(tensor, length):
    to_pad = length - len(tensor)
    pad = Variable(torch.zeros(to_pad).long())
    if torch.cuda.is_available():
        pad = pad.cuda()
    return torch.cat([tensor, pad])

def scenes_to_vec(scenes):
    max_words = max(len(scene.description) for scene in scenes) - 1
    word_data = Variable(torch.zeros(len(scenes), max_words))

    if torch.cuda.is_available():
        word_data = word_data.cuda()

    for i_scene, scene in enumerate(scenes):
        offset = max_words - len(scene.description)
        for i_word, word in enumerate(scene.description):
            if word != EOS:
                word_data[i_scene, i_word] = word

    word_data = word_data.long()
    return word_data

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)

class Listener0Model(nn.Module):
    def __init__(self, vocab_sz, num_scenes, hidden_sz, output_sz, dropout): #figure out what parameters later
        super(Listener0Model, self).__init__()
        self.vocab_sz = vocab_sz
        self.num_scenes = num_scenes
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.dropout_p = dropout # need to pass this in somewhere
        self.name='Listener0'

        self.scene_input_sz = N_PROP_TYPES * N_PROP_OBJECTS

        self.scene_encoder = LinearSceneEncoder("Listener0", self.scene_input_sz, hidden_sz, dropout) #figure out what parameters later
        self.string_encoder = LinearStringEncoder("Listener0", vocab_sz, hidden_sz, dropout) #figure out what parameters later
        self.scorer = MLPScorer("Listener0", hidden_sz, output_sz, dropout) #figure out what parameters later
        # self.fc = nn.Linear() #Insert something here what is this?


    def forward(self, data, alt_data): # alt_data seems to be a list, data seems to have both string and image
        scene_enc = self.scene_encoder(data)                            # [100 x 50]
        alt_scene_enc = [self.scene_encoder(alt) for alt in alt_data]   # [100 x 50]

        string_enc = self.string_encoder(data)                          # [100 x 50]
        scenes = [scene_enc] + alt_scene_enc                            # List of length 2
        labels = torch.zeros((len(data),))                              # length 100
        log_probs = self.scorer(string_enc, scenes, labels)

        return log_probs

class Speaker0Model(nn.Module):
    def __init__(self, vocab_sz, hidden_sz, dropout, string_decoder='LSTM'): #figure out what parameters later
        super(Speaker0Model, self).__init__()

        self.name='Speaker0'

        self.vocab_sz = vocab_sz
        self.hidden_sz = hidden_sz
        self.scene_input_sz = N_PROP_OBJECTS * N_PROP_TYPES

        self.scene_encoder = LinearSceneEncoder("Speaker0SceneEncoder", self.scene_input_sz, hidden_sz, dropout)

        embedding_dim = self.hidden_sz # ???

        if string_decoder == 'LSTM':
            self.string_decoder = LSTMStringDecoder("Speaker0StringDecoder", self.vocab_sz, embedding_dim, self.hidden_sz, dropout)
        else:
            self.string_decoder = MLPStringDecoder("Speaker0StringDecoder", self.hidden_sz, self.vocab_sz, dropout) # Not sure what the input and hidden size are for this
        
        # name, vocab_sz, embedding_dim, hidden_sz, dropout, num_layers=2):
        # self.fc = nn.Linear() #Insert something here Why is this needed?

        self.dropout_p = dropout

    def forward(self, data, alt_data):
        scene_enc = self.scene_encoder(data)
        max_len = max(len(scene.description) for scene in data)
        losses = self.string_decoder(scene_enc, data, max_len) # losses was [1400 x 2713]
        # should probs be like
        return losses

    def sample(self, data, alt_data, viterbi=False, k=10):
        scene_enc = self.scene_encoder(data) # [100 x 50]
        max_len = max(len(scene.description) for scene in data) # 15
        probs, sampled_ids = self.string_decoder.sample(scene_enc, max_len, viterbi, k=k) # used to return probs, sample
        return probs, sampled_ids # used to return probs, np.zeros(probs.shape), sample


class CompiledSpeaker1Model(nn.Module):
    def __init__(self, vocab_sz, num_scenes, hidden_sz, output_sz, dropout): #figure out what parameters later
        super(CompiledSpeaker1Model, self).__init__()
        self.vocab_sz = vocab_sz
        self.num_scenes = num_scenes
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz

        self.scene_input_sz = N_PROP_TYPES * N_PROP_OBJECTS

        self.sampler = SamplingSpeaker1Model(vocab_sz, num_scenes, hidden_sz, output_sz, dropout) # send params
        self.scene_encoder = LinearSceneEncoder("CompSpeaker1Model", self.scene_input_sz, hidden_sz, dropout)
        self.string_decoder = MLPStringDecoder("CompSpeaker1Model", hidden_sz, vocab_sz, dropout)

        # self.fc = nn.Linear()
        self.dropout_p = dropout

    def forward(self, data, alt_data):
        _, _, samples = self.sampler.sample(data, alt_data)

        scene_enc = self.scene_encoder(data)
        alt_scene_enc = [self.scene_encoder(alt) for alt in alt_data]

        scenes = [scene_enc] + alt_scene_enc

        fake_data = [d._replace(description=s) for d, s in zip(data, samples)]
        losses = self.string_decoder(fake_data)
        return losses, np.asarray(0)

    def sample(self, data, alt_data, viterbi, quantile=None):
        scene_enc = self.scene_encoder.forward("true", data, self.dropout_p)
        alt_scene_enc = [self.scene_encoder.forward("alt%d" % i, alt, self.dropout_p)
                            for i, alt in enumerate(alt_data)]
        ### figure out how to translate these lines
        l_cat = "CompSpeaker1Model_concat"
        self.apollo_net.f(Concat(
            l_cat, bottoms=[scene_enc] + alt_scene_enc))
        ###

        probs, sample = self.string_decoder.sample("", l_cat, viterbi)
        return probs, np.zeros(probs.shape), sample

def tensor_to_caption(tensor):
    caption = []
    for word_id in tensor:
        word = WORD_INDEX.get(word_id.data[0])
        caption.append(word)
        if word_id.data[0] == EOS:
            break
    return (' '.join(caption))


class SamplingSpeaker1Model(nn.Module):
    def __init__(self, listener0, speaker0): #figure out what parameters later
        super(SamplingSpeaker1Model, self).__init__()

        # self.listener0 = Listener0Model(vocab_sz, num_scenes, hidden_sz, output_sz, dropout)
        # self.speaker0 = Speaker0Model(vocab_sz, hidden_sz, dropout)
        self.listener0 = listener0
        self.speaker0 = speaker0
        self.name = 'SamplingSpeaker1'

    def sample(self, data, alt_data, viterbi=None, quantile=None, k=10):
        n_samples = k 

        speaker_scores = []
        listener_scores = []

        all_speaker_log_probs, all_sampled_ids = self.speaker0.sample(data, alt_data, viterbi=False, k=n_samples) # used to output [speaker_log_probs, _, sample]
        # all_sampled_ids is  # [100 x  k x 20]
        # all_speaker_log_probs = [100 x k]

        def create_fake_scenes(fake_description_ids, original_scene):
            # fake descr = [k x 20]
            # original scene = Scene(blahblah)
            # fake_descriptions = [tensor_to_caption(ids) for ids in fake_description_ids]
            fake_scenes = [original_scene._replace(description=caption.data) for caption in fake_description_ids]
            return fake_scenes

        def select_best_description(scores, fake_description_ids):
            # Scores should be [k x 2]
            scores_for_correct = scores[:, 0]
            value, ind = scores_for_correct.max(dim=0) # 
            return fake_description_ids[ind]

        # Lambda trades off between L0 and S0. This is joint probability of sentence by both listener and speaker
        def select_best_weighted_description(listener_scores, speaker_scores, fake_description_ids, lam): # lambda controls weighing of speaker to listener
            scores_for_correct = (listener_scores[: 0] ** (1-lam)) * (speaker_scores[: 0] ** lam) # Check this is element-wise mult
            value, ind = scores_for_correct.max(dim=0)
            return fake_description_ids[ind]

        ids_split = torch.unbind(all_sampled_ids, dim=0) # tuple, each of which is [10 x 20]
        all_fake_scenes = [create_fake_scenes(fake_description_ids, original_scene) for fake_description_ids, original_scene in zip(ids_split, data)]

        all_listener_log_probs = [self.listener0(fake_scenes, [[alt_data[0][i]] * n_samples]) for i, fake_scenes in enumerate(all_fake_scenes)]

        # listener_scores = torch.stack(listener_scores, 2)


        best_descriptions = [select_best_description(scores, fake_description_ids) for scores, fake_description_ids in zip(all_listener_log_probs, ids_split)]

        out_descriptions = torch.stack(best_descriptions)
        pdb.set_trace()
        return (all_listener_log_probs, all_speaker_log_probs), out_descriptions


        # speaker_scores = torch.stack(speaker_scores, 2)         # [100 x 20 x 10] , 20 from max sample length
        # listener_scores = torch.stack(listener_scores, 2)       # [100 x 2 x 10]

        # stacked_sentences = Variable(torch.stack(out_sentences)) # [100 x 20]
        # return (out_speaker_scores, out_listener_scores), stacked_sentences

class LinearStringEncoder(nn.Module):
    def __init__(self, name, vocab_sz, hidden_sz, dropout): #figure out what parameters later
        super(LinearStringEncoder, self).__init__()
        self.name = name
        self.vocab_sz = vocab_sz
        self.hidden_sz = hidden_sz
        self.fc = nn.Linear(vocab_sz, hidden_sz)
        self.dropout = dropout

    def forward(self, scenes):
        feature_data = Variable(torch.zeros(len(scenes), self.vocab_sz))    # [100 x vocab_sz]
        if torch.cuda.is_available():
            feature_data = feature_data.cuda()

        for i_scene, scene in enumerate(scenes):
            for word in scene.description:
                # feature_data[i_scene, word.data[0]] = feature_data[i_scene, word.data[0]] + 1
                feature_data[i_scene, word] = feature_data[i_scene, word] + 1
        # logging.info("LinearStringEncoder_" + prefix)
        # logging.info("LinearStringEncoder_")

        result = self.fc(feature_data)
        return result

class LinearSceneEncoder(nn.Module):
    def __init__(self, name, input_sz, hidden_sz, dropout): #figure out what parameters later
        super(LinearSceneEncoder, self).__init__()
        self.name = name
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.fc = nn.Linear(input_sz, hidden_sz)
        self.dropout_p = dropout

    def forward(self, scenes):
        feature_data = Variable(torch.zeros(len(scenes), N_PROP_TYPES * N_PROP_OBJECTS))
        if torch.cuda.is_available():
            feature_data = feature_data.cuda()

        for i_scene, scene in enumerate(scenes):
            for prop in scene.props:
                feature_data[i_scene, prop.type_index * N_PROP_OBJECTS +
                        prop.object_index] = 1
        # logging.info("LinearSceneEncoder_" + prefix)
        # logging.info("LinearSceneEncoder_")
        result = self.fc(feature_data)
        return result

class LSTMStringDecoder(nn.Module):
    def __init__(self, name, vocab_sz, embedding_dim, hidden_sz, dropout, num_layers=2): #figure out what parameters later
        super(LSTMStringDecoder, self).__init__()
        self.name = name
        self.vocab_sz = vocab_sz
        self.embedding_dim = embedding_dim
        self.hidden_sz = hidden_sz
        self.num_layers = num_layers

        self.dropout_p = dropout

        self.embedding = nn.Embedding(vocab_sz, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_sz, num_layers, dropout=self.dropout_p, batch_first=True)
        self.linear = nn.Linear(hidden_sz, vocab_sz)
        self.dropout = nn.Dropout(self.dropout_p)

        self.init_param = 1.0
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.init_param, self.init_param)
        self.linear.weight.data.uniform_(-self.init_param, self.init_param)

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)).cuda(),
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)).cuda())

        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)),
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)))

    def forward(self, scene_enc, scenes, max_words):
        batch_size = len(scene_enc) # [100 x 50]
        word_data = scenes_to_vec(scenes) # [100 x 15]

        hidden = self.init_hidden(batch_size)
        embedding = self.embedding(word_data) # dimensions = [100 x 15 x 50]
        embedding = torch.cat((scene_enc.unsqueeze(1), embedding), 1) # after: [100 x 16 x 50]?
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output) # [100 x 15 x 50]
        output = self.linear(output.view(-1, self.hidden_sz)) # -> [1400 x 50] (batch, max_words x hidden_size) # [1400 x 2713] (to vocab size?
        return output


    def forward_step(self, source, last_hidden, encoder_outputs, use_embeddings=True):
        word_embeddings = self.embedding(source) if use_embeddings else source
        output, new_hidden = self.lstm(word_embeddings, last_hidden)
        output = self.dropout(output)
        output = self.linear(output.view(-1, self.hidden_sz)) 
        return output, new_hidden

    def sample(self, scene_enc, max_words, viterbi, k=10):
        self.eval()
        samples = []
        probs = []
        for scene in scene_enc:
            prob, sample = self.beam_sample(scene.unsqueeze(1), max_words, viterbi, k=k)
            samples.append(sample)
            probs.append(prob)
        return torch.stack(probs), torch.stack(samples) # [100 x  5 x 21]


    def beam_sample(self, scene_enc, max_words, viterbi, k=10):
        batch_size = 1
        encoder_outputs = None
        # Change scene_enc to be [1 x 1 x 50]
        scene_enc = scene_enc.transpose(0,1).unsqueeze(0)
        initial_guess =  Variable(torch.LongTensor([SOS]).view(1, 1)) # Or should this be SOS?? fix later
        initial_hidden = self.init_hidden(batch_size)# [2 x 1 x 50] or maybe scene_enc??
        zeroth, decoder_hidden = self.forward_step(scene_enc, initial_hidden, encoder_outputs, use_embeddings=False)
        if torch.cuda.is_available():
            initial_guess = initial_guess.cuda()
        current_hypotheses = [(0, initial_guess, decoder_hidden)] # (prob, )

        completed_guesses = []

        for i in range(MAX_LEN - 1):
            guesses_for_this_length = []
            while (current_hypotheses != []):
                # Pop something off the current hypotheses
                log_prob, last_sequence_guess, decoder_hidden = current_hypotheses.pop(0)

                last_word = last_sequence_guess[-1:, :]
                # EOS token:
                if last_word.squeeze().data[0] == EOS: 
                    completed_guesses.append((log_prob, last_sequence_guess, None))
                else:
                    decoder_outputs, decoder_hidden = self.forward_step(last_word, decoder_hidden, encoder_outputs) # target, last_hidden, encoder_outputs
                    # # Get k hypotheses for each 
                    # decoder outputs is [batch x target_len x en_vocab_sz]  = [ 1 x 1 x vocab]
                    vocab_size = self.vocab_sz
                    n_probs, n_indices = torch.topk(decoder_outputs, k, dim=1) 
                    new_probs = F.log_softmax(n_probs, dim=1) + log_prob# this should be tensor of size k 
                    new_probs = new_probs.squeeze().data
                    new_sequences = [torch.cat([last_sequence_guess, n_index.view(1, 1)],dim=0) for n_index in n_indices.squeeze()] # check this
                    new_hidden = [decoder_hidden] * k
                    # decoder_hidden: # tuple, each of which is [num_layers x batch x hidden]
                    seq_w_probs = list(zip(new_probs, new_sequences, new_hidden))
                    guesses_for_this_length = guesses_for_this_length + seq_w_probs

            # Top k current hypotheses after this time step:
            guesses_for_this_length = sorted(guesses_for_this_length, key= lambda tup: -1*tup[0])[:k]

            current_hypotheses = current_hypotheses + guesses_for_this_length

        # Return top result
        completed_guesses = completed_guesses + guesses_for_this_length

        completed_guesses.sort(key= lambda tup: -1*tup[0])
        completed_guesses = completed_guesses[:k]
        sentences =  [x[1].squeeze() for x in completed_guesses]
        probs = [x[0] for x in completed_guesses]

        sentences = [pad_end1d(tensor, MAX_LEN) for tensor in sentences]


        return torch.Tensor(probs), torch.stack(sentences) # [5 x 20] tensor




    # Currently performs a greedy search
    def old_sample(self, scene_enc, max_words, viterbi):
        pdb.set_trace()
        batch_size = len(scene_enc)                     # 100
        sampled_ids = []
        out_log_probs = []

        hidden = self.init_hidden(batch_size)           # [2 x 100 x 50]
        inputs = scene_enc.unsqueeze(1)                 # [100 x 1 x 50]
        for i in range(20):                             # maximum sampling length
            output, hidden = self.lstm(inputs, hidden)  # [100 x 1 x 50], [2 x 100 x 50]
            output = self.linear(output.squeeze(1))     # [100 x 2713] (vocab size)

            # need to figure out if I need to check if predicted had 2 (end of sentence) in it.
            out_prob, predicted = output.max(1) # predicted is size [100]
            out_log_probs.append(torch.log(out_prob))
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        out_log_probs = torch.stack(out_log_probs, 1)
        return out_log_probs, sampled_ids

class MLPScorer(nn.Module):
    def __init__(self, name, hidden_sz, output_sz, dropout): #figure out what parameters later
        super(MLPScorer, self).__init__()
        self.name = name
        self.output_sz = output_sz
        self.hidden_sz = hidden_sz # hidden_sz refers to the encoding size?
        self.dropout_p = dropout

        self.intermediate_sz = hidden_sz # not sure..

        self.linear_4 = nn.Linear(hidden_sz, self.intermediate_sz) # Referent (scene) encodings
        self.linear_5 = nn.Linear(hidden_sz, self.intermediate_sz) # String encodings
        self.linear_3 = nn.Linear(self.intermediate_sz, 1) # what size is this supposed to be?


    def forward(self, query, targets, labels): # string_enc, scenes, labels
        # targets = scenes? each is [100 x 50] = batch_size x hidden_sz -> 
        num_targets = len(targets) # 2 

        targets_after_linear = [self.linear_4(target).unsqueeze(1) for target in targets]
        targets = torch.cat(targets_after_linear, dim=1)

        string_enc = self.linear_5(query).unsqueeze(1) # w_5 * e_d

        linear_combination = targets + string_enc # batch_sz x 2 x output?

        post_relu = F.relu(linear_combination)

        ss = self.linear_3(post_relu).squeeze() # [batch_size x 2] after squeeze

        return ss

        # should we output the log softmaxes???
        return F.log_softmax(ss, dim=1).squeeze() #i guess not for cross entropy

        # # query.unsqueeze_(1) # should be batch_sz, 1, n_dims = 50 (hidden size)
        # new_query = query.expand(-1, num_targets)
        # new_sum = new_query + targets # element wise summation. MAY NOT WORK

        # result = self.linear(new_sum).squeeze(1) # should now be batch_sz, n_dims
        # return result

class MLPStringDecoder(nn.Module):
    def __init__(self, name, hidden_sz, vocab_sz, dropout):
        super(MLPStringDecoder, self).__init__()
        self.vocab_sz = vocab_sz
        self.forward_net = nn.Sequential(
            nn.Linear(2 * vocab_sz + hidden_sz, vocab_sz), # Linear 7
            nn.ReLU(),
            nn.Linear(vocab_sz, vocab_sz)
            )
        self.max_words = 20 #?????

    def one_hot(self, batch, depth):
        if torch.cuda.is_available():
            batch = batch.cpu()
        ones = torch.sparse.torch.eye(depth)
        return ones.index_select(0,batch)

    def forward_step(self, d_n, d_prev, e_r):
        # d_n = indicator feature on previous word, should be of size vocab_sz?
        # d_prev: also vocab_sz, indicator feature on all previous (basically BOW)
        # e_r: hidden_sz
        inp = torch.cat([d_n, d_prev, e_r], dim=1) # result has size batch_sz x [2 * vocab + hidden]
        return self.forward_net(inp) # Log softmax or not?? result rn has size [100 x 2713]

    def forward(self, scene_enc, targets, max_words): # Input is image encoding
        # Input is scene_enc: [batch_sz x hidden_sz]
        # max_words = self.max_words
        batch_sz = len(scene_enc)

        start_of_sentence = torch.ones(batch_sz).long() # ones to signal <s> [batch_sz]
        d_n = Variable(self.one_hot(start_of_sentence, self.vocab_sz)) # [batch_sz x vocab_sz]
        d_prev = Variable(torch.zeros(batch_sz, self.vocab_sz)) # [batch_sz x vocab_sz]

        losses = []

        for i in range(1, max_words):
            if torch.cuda.is_available():
                d_n, d_prev = d_n.cuda(), d_prev.cuda()

            out = self.forward_step(d_n, d_prev, scene_enc) # [batch_sz x vocab_sz]
            losses.append(out)
            values, indices = torch.max(out, 1)
            d_prev += d_n # Add last word to d_prev
            d_n = Variable(self.one_hot(indices.data, self.vocab_sz))

        output = torch.cat(losses, dim=0) # [(batch_sz * len) x vocab_sz]
        return output

