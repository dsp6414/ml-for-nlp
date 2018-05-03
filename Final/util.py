import pdb
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch
import logging

MAX_LEN = 20
SOS = 1
EOS = 2
class Struct:
    def __init__(self, **entries):
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = Struct(**v)
            elif isinstance(v, list):
                rv = []
                for item in v:
                    if isinstance(item, dict):
                        rv.append(Struct(**item))
                    else:
                        rv.append(item)
            else:
                rv = v
            rec_entries[k] = rv
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, Struct):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "struct {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "Struct(%r)" % self.__dict__

class Index:
    def __init__(self):
        self.contents = dict()
        self.ordered_contents = []
        self.reverse_contents = dict()

    def __getitem__(self, item):
        if item not in self.contents:
            return None
        return self.contents[item]

    def index(self, item):
        if item not in self.contents:
            idx = len(self.contents) + 1
            self.ordered_contents.append(item)
            self.contents[item] = idx
            self.reverse_contents[idx] = item
        idx = self[item]
        assert idx != 0
        return idx

    def get(self, idx):
        if idx == 0:
            return "*invalid*"
        return self.reverse_contents[idx]

    def __len__(self):
        return len(self.contents) + 1

    def __iter__(self):
        return iter(self.ordered_contents)

def flatten(lol):
    if isinstance(lol, tuple) or isinstance(lol, list):
        return sum([flatten(l) for l in lol], [])
    else:
        return [lol]

def postorder(tree):
    if isinstance(tree, tuple):
        for subtree in tree[1:]:
            for node in postorder(subtree):
                yield node
        yield tree[0]
    else:
        yield tree

def tree_map(function, tree):
    if isinstance(tree, tuple):
        head = function(tree)
        tail = tuple(tree_map(function, subtree) for subtree in tree[1:])
        return (head,) + tail
    return function(tree)

def tree_zip(*trees):
    if isinstance(trees[0], tuple):
        zipped_children = [[t[i] for t in trees] for i in range(len(trees[0]))]
        zipped_children_rec = [tree_zip(*z) for z in zipped_children]
        return tuple(zipped_children_rec)
    return trees

# listener (all zeros because the correct choice is the 0th one)
def listener_targets(args, scenes):
    return Variable(torch.zeros(args.batch_size)).long()

# speaker (this is annoying)
def speaker0_targets(args, scenes):
    max_words = max(len(scene.description) for scene in scenes)

    targets = Variable(torch.zeros(len(scenes), max_words)) # [100 x 15]
    for i_scene, scene in enumerate(scenes):
        offset = max_words - len(scene.description)
        for i_word, word in enumerate(scene.description):
            targets[i_scene, i_word] = word

    targets = targets.view(-1).long()
    return targets


def train(train_scenes, model, optimizer, args, target_func):
    n_train = len(train_scenes) 
    model.train()

    criterion = nn.CrossEntropyLoss() 

    n_train_batches = int(n_train / args.batch_size) # just truncate this i guess

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0

        for i_batch in range(n_train_batches):
            optimizer.zero_grad()
            batch_data = train_scenes[i_batch * args.batch_size : 
                                      (i_batch + 1) * args.batch_size]
            alt_indices = \
                    [np.random.choice(n_train, size=args.batch_size)
                     for i_alt in range(args.alternatives)]
            alt_data = [[train_scenes[i] for i in alt] for alt in alt_indices]
            
            outputs = model(batch_data, alt_data)
            targets = target_func(args, batch_data)

            if torch.cuda.is_available():
                targets = targets.cuda()

            loss = criterion(outputs, targets) # what should these be?
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

            if (i_batch % args.log_interval == 0):
                logging.info('Epoch [%d/%d], Step[%d/%d], loss: %.4f' 
                  %(epoch, args.epochs, i_batch, n_train_batches, loss.data[0]))

        logging.info('====> Epoch %d: Training loss: %.4f' % (epoch, epoch_loss))


def sample(decoder, scene_enc): # Predict with beam search based on scene enc
    initial_guess = scene_enc
    decoder_hidden = decoder.init_hidden()# [SOMETHING] or maybe scene_enc??
    if torch.cuda.is_available():
        initial_guess = initial_guess.cuda()
    current_hypotheses = [(0, initial_guess, decoder_hidden)] # 

    completed_guesses = []

    for i in range(MAX_LEN):
        guesses_for_this_length = []
        while (current_hypotheses != []):
            # Pop something off the current hypotheses
            hypothesis = current_hypotheses.pop(0)
            log_prob, last_sequence_guess, decoder_hidden = hypothesis
            
            last_word = last_sequence_guess[-1:, :]
            # EOS token:
            if last_word.squeeze().data[0] == EOS: 
                completed_guesses.append((log_prob, last_sequence_guess, None))
            else:
                decoder_outputs, decoder_hidden = decoder(last_word, decoder_hidden, encoder_outputs)
                # # Get k hypotheses for each 
                # decoder outputs is [target_len x batch x en_vocab_sz] -> [1 x 1 x vocab]
                vocab_size = len(decoder_outputs[0][0])
                n_probs, n_indices = torch.topk(decoder_outputs, k, dim=2)
                new_probs = F.log_softmax(n_probs, dim=2) + log_prob# this should be tensor of size k 
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
    return [x[1] for x in completed_guesses]

def get_examples(model, train_scenes, args):
    pdb.set_trace()
    model.eval()
    n_train = len(train_scenes) 
    n_train_batches = int(n_train / args.batch_size)

    for i_batch in range(n_train_batches):
        batch_data = train_scenes[i_batch * args.batch_size : 
                                      (i_batch + 1) * args.batch_size]
        alt_indices = \
                [np.random.choice(n_train, size=args.batch_size)
                 for i_alt in range(args.alternatives)]
        alt_data = [[train_scenes[i] for i in alt] for alt in alt_indices]

        pdb.set_trace()
        model.sample(batch_data, alt_data)





def setup_logging(args):
    with open("log/file_num.txt") as file: 
        experiment_counter = file.read()

    with open("log/file_num.txt", 'w') as file: 
        file.write(str(int(experiment_counter) + 1))

    log_prefix = args.model if args.model else 'exp'
    log_file = args.model + '_' + experiment_counter +'.out'
    log_path = 'log/' + log_file
    level = logging.INFO 
    format = ' %(message)s' 
    handlers = [logging.FileHandler(log_path), logging.StreamHandler()] 
    logging.basicConfig(level = level, format = format, handlers = handlers) 
