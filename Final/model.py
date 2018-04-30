import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

N_PROP_TYPES = 8
N_PROP_OBJECTS = 35

class Listener0Model(nn.Module):
    def __init__(self, vocab_sz, num_scenes, hidden_sz, output_sz, dropout): #figure out what parameters later
        super(Listener0Model, self).__init__()
        self.vocab_sz = vocab_sz
        self.num_scenes = num_scenes
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.dropout_p = dropout # need to pass this in somewhere

        self.scene_input_sz = N_PROP_TYPES * N_PROP_OBJECTS

        self.scene_encoder = LinearSceneEncoder("Listener0", self.scene_input_sz, hidden_sz) #figure out what parameters later
        self.string_encoder = LinearStringEncoder("Listener0", vocab_sz, hidden_sz) #figure out what parameters later
        self.scorer = MLPScorer("Listener0", hidden_sz, output_sz, dropout) #figure out what parameters later
        # self.fc = nn.Linear() #Insert something here what is this?


    def forward(self, data, alt_data): # alt_data seems to be a list, data seems to have both string and image
        scene_enc = self.scene_encoder(data)
        alt_scene_enc = [self.scene_encoder(alt_data) for i in alt_data]
        string_enc = self.string_encoder(data) # data has the string?

        scenes = [scene_enc] + alt_scene_enc
        labels = np.zeros((len(data),))
        log_probs, accs = self.scorer(scenes, string_enc, labels)

        return log_probs, accs

class Speaker0Model(nn.Module):
    def __init__(self, vocab_sz, hidden_sz, dropout): #figure out what parameters later
        super(Speaker0Model, self).__init__()

        self.vocab_sz = vocab_sz
        self.hidden_sz = hidden_sz
        self.scene_input_sz = N_PROP_OBJECTS * N_PROP_TYPES

        self.scene_encoder = LinearSceneEncoder("Speaker0SceneEncoder", self.scene_input_sz, hidden_sz)
        self.string_decoder = MLPStringDecoder("Speaker0StringDecoder", self.hidden_sz, self.hidden_sz, self.vocab_sz, dropout) # Not sure what the input and hidden size are for this
        # self.fc = nn.Linear() #Insert something here Why is this needed?

        self.dropout_p = dropout

    def forward(self, data, alt_data):
        scene_enc = self.scene_encoder(data)
        losses = self.string_decoder("", scene_enc, data) # this seems off. no calling alt_data?

        return losses, np.asarray(0)

    # I have no idea what's going on here
    def sample(self, data, alt_data, viterbi, quantile=None):
        scene_enc = self.scene_encoder("", data)
        probs, sample = self.string_decoder("", scene_enc, viterbi)
        return probs, np.zeros(probs.shape), sample

class SamplingSpeaker1Model(nn.Module):
    def __init__(self, vocab_sz, num_scenes, hidden_sz, output_sz, dropout): #figure out what parameters later
        super(SamplingSpeaker1Model, self).__init__()

        self.listener0 = Listener0Model(vocab_sz, num_scenes, hidden_sz, output_sz, dropout)
        self.speaker0 = Speaker0Model(vocab_sz, hidden_sz, dropout)

        # self.fc = nn.Linear() # figure out parameters

    def sample(self, data, alt_data, viterbi, quantile=None):
        if viterbi or quantile is not None:
            n_samples = 10
        else:
            n_samples = 1

        speaker_scores = np.zeros((len(data), n_samples))
        listener_scores = np.zeros((len(data), n_samples))

        all_fake_scenes = []
        for i_sample in range(n_samples):
            speaker_log_probs, _, sample = self.speaker0.sample(data, alt_data, dropout, viterbi=False)

            fake_scenes = []
            for i in range(len(data)):
                fake_scenes.append(data[i]._replace(fake_scenes, alt_data, dropout)) # do I need dropout here
            all_fake_scenes.append(fake_scenes)

            listener_logprobs, accs = self.listener0.forward(fake_scenes, alt_data, dropout) # dropout"
            speaker_scores[:, i_sample] = speaker_log_probs
            listener_scores[:, i_sample] = listener_log_probs

        scores = listener_scores

        out_sentences = []
        out_speaker_scores = np.zeros(len(data))
        out_listener_scores = np.zeros(len(data))

        for i in range(len(data)):
            if viterbi:
                q = scores[i, :].argmax()
            elif quantile is not None:
                idx = int(n_samples * quantile)
                if idx == n_samples:
                    q = scores.argmax()
                else:
                    q = scores[i,:].argsort()[idx]
            else:
                q = 0
            out_sentences.append(all_fake_scenes[q][i].description)
            out_speaker_scores[i] = speaker_scores[i][q]
            out_listener_scores[i] = listener_scores[i][q]

        return out_speaker_scores, out_listener_scores, out_sentences

class CompiledSpeaker1Model(nn.Module):
    def __init__(self, vocab_sz, hidden_sz, dropout): #figure out what parameters later
        super(CompiledSpeaker1Model, self).__init__()
        self.vocab_sz = vocab_sz
        self.hidden_sz = hidden_sz

        self.scene_input_sz = N_PROP_TYPES * N_PROP_OBJECTS

        self.sampler = SamplingSpeaker1Model() # send params
        self.scene_encoder = LinearSceneEncoder("CompSpeaker1Model", self.scene_input_sz, hidden_sz)
        self.string_decoder = MLPStringDecoder("CompSpeaker1Model")

        self.fc = nn.Linear() # maybe??
        self.dropout_p = dropout

    def forward(self, data, alt_data):
        _, _, samples = self.sampler.sample(data, alt_data, self.dropout_p, True)

        scene_enc = self.scene_encoder.forward("true", data, self.dropout_p)
        alt_scene_enc = [self.scene_encoder.forward("alt%d" % i, alt, self.dropout_p)
                            for i, alt in enumerate(alt_data)]

        ### figure out how to translate these lines
        l_cat = "CompSpeaker1Model_concat"
        self.apollo_net.f(Concat(
            l_cat, bottoms=[scene_enc] + alt_scene_enc))
        ###

        fake_data = [d._replace(description=s) for d, s in zip(data, samples)]

        losses = self.string_decoder.forward("", l_cat, fake_data, self.dropout_p)
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

class LinearStringEncoder(nn.Module):
    def __init__(self, name, vocab_sz, hidden_sz): #figure out what parameters later
        super(LinearStringEncoder, self).__init__()
        self.name = name
        self.vocab_sz = vocab_sz
        self.hidden_sz = hidden_sz
        self.fc = nn.Linear(vocab_sz, hidden_sz)

    def forward(self, prefix, scenes, dropout):
        feature_data = Variable(torch.zeros(len(scenes), self.vocab_sz))
        if torch.cuda.is_available():
            feature_data = feature_data.cuda()

        for i_scene, scene in enumerate(scenes):
            for word in scene.description:
                feature_data[i_scene, word] += 1
        print("LinearStringEncoder_" + prefix)
        result = self.fc(feature_data)
        return result

class LinearSceneEncoder(nn.Module):
    def __init__(self, name, input_sz, hidden_sz): #figure out what parameters later
        super(LinearSceneEncoder, self).__init__()
        self.name = name
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.fc = nn.Linear(input_sz, hidden_sz)

    def forward(self, prefix, scenes, dropout):
        feature_data = Variable(torch.zeros(len(scenes), N_PROP_TYPES * N_PROP_OBJECTS))
        if torch.cuda.is_available():
            feature_data = feature_data.cuda()

        for i_scene, scene in enumerate(scenes):
            for prop in scene.props:
                feature_data[i_scene, prop.type_index * N_PROP_OBJECTS +
                        prop.object_index] = 1
        print("LinearSceneEncoder_" + prefix)
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
        self.lstm = nn.LSTM(embedding_dim, hidden_sz, num_layers, dropout=self.dropout_p)
        self.linear = nn.Linear(hidden_sz, vocab_sz)
        self.dropout = nn.Dropout(self.dropout_p)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.init_param, self.init_param)
        self.linear.weight.data.uniform_(-self.init_param, self.init_param)

    def init_hidden(self, batch_size=100):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)).cuda(),
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)).cuda())

        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)),
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_sz)))

    def forward(self, prefix, encoding, scenes): # why do you need encoding or prefix?
        max_words = max(len(scene.description) for scene in scenes)
        word_data = Variable(torch.zeros(len(scenes), max_words))

        if torch.cuda.is_available():
            word_data = word_data.cuda()

        for i_scene, scene in enumerate(scenes):
            offset = max_words - len(scene.description)
            for i_word, word in enumerate(scene.description):
                word_data[i_scene, i_word] = word

        print("LSTMStringDecoder_" + prefix)

        hidden = init_hidden()
        embedding = self.embedding(word_data) # find out dimensions of word_data
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        output = self.linear(output.view(-1, self.hidden_sz))
        return output, hidden

class MLPScorer(nn.Module):
    def __init__(self, name, hidden_sz, output_sz, dropout): #figure out what parameters later
        super(MLPScorer, self).__init__()
        self.name = name
        self.output_sz = output_sz
        self.hidden_sz = hidden_sz
        self.dropout_p = dropout

        self.linear = nn.Linear(hidden_sz, output_sz)

    def forward(self, prefix, query, targets):
        print("MLPScorer_" + prefix)
        
        num_targets = len(targets)
        for target in targets:
            target.unsqueeze_(1) # should be batch_sz, 1, n_dims = 50 (hidden size)
        query.unsqueeze_(1) # should be batch_sz, 1, n_dims = 50 (hidden size)
        targets = torch.cat(targets, dim=1)
        new_query = query.expand(-1, num_targets)
        new_sum = new_query + targets # element wise summation. MAY NOT WORK

        result = self.linear(new_sum).squeeze(1) # should now be batch_sz, n_dims
        return result

class MLPStringDecoder(nn.Module):
    def __init__(self, name, input_sz, hidden_sz, vocab_sz, dropout):
        super(MLPStringDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_sz, hidden_sz),
            nn.Linear(hidden_sz, hidden_sz),
            nn.Linear(hidden_sz, vocab_sz),
            nn.Dropout(dropout)
            )

    def forward(self, scenes):
        max_words = max(len(scene.description) for scene in scenes)

        word_data = Variable(torch.zeros(len(scenes), max_words))

        if torch.cuda.is_available():
            word_data = word_data.cuda()

        for i_scene, scene in enumerate(scenes):
            offset = max_words - len(scene.description)
            for i_word, word in enumerate(scene.description):
                word_data[i_scene, i_word] = word

        output = self.net(word_data)
        return output




