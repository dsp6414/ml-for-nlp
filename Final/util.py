import os
import sys
from collections import defaultdict
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch
import logging
import nltk
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image, ImageDraw

import pdb

MAX_LEN = 20
SOS = 1
EOS = 2

torch.manual_seed(1)

data_path = 'AbstractScenes_v1.1/' if os.path.exists('AbstractScenes_v1.1/') else '../../../../../AbstractScenes_v1.1/'

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


# listener (all zeros because the correct choice is the 0th one)
def listener_targets(args, scenes):
    return Variable(torch.zeros(args.batch_size)).long()

# speaker 
def speaker0_targets(args, scenes):
    max_words = max(len(scene.description) for scene in scenes)

    targets = Variable(torch.zeros(len(scenes), max_words)) # [100 x 15]
    for i_scene, scene in enumerate(scenes):
        offset = max_words - len(scene.description)
        for i_word, word in enumerate(scene.description):
            targets[i_scene, i_word] = word

    targets = targets.view(-1).long()
    return targets

def print_tensor_1d(data, WORD_INDEX):
    logging.info([WORD_INDEX.get(word) for word in data])
def print_tensor(data, WORD_INDEX):
    for x in data:
        logging.info([WORD_INDEX.get(word) for word in x])

def print_tensor3d(data, WORD_INDEX):
    for ind, x in enumerate(data):
        logging.info('Training row %d'  % (ind))
        for y in x:
            logging.info([WORD_INDEX.get(word) for word in y])

def print_datas_and_desc(data, alt_data, sentences, WORD_INDEX):
    for scene, alt_scenes, sent in zip(data, alt_data, sentences):
        logging.info('Scene ID: %s, alt scene id: %s' % (scene.image_id, alt_scenes[0].image_id))
        print_tensor(sent, WORD_INDEX)

def tensor_to_caption(s, WORD_INDEX):
    index_of_end = (s == 2).nonzero()[0][0] if len((s == 2).nonzero()) > 0 else MAX_LEN
    s_chopped = s[1:index_of_end.data[0]] if index_of_end.data[0] > 1 else []
    s_joined = ' '.join([WORD_INDEX.get(i.data[0]) for i in s_chopped])
    return s_joined

def validate(val_scenes, model, optimizer, args, target_func, epoch):
    model.eval()
    epoch_loss = 0.0
    total_correct = 0.0
    criterion = nn.CrossEntropyLoss()
    n_val = len(val_scenes) 
    n_val_batches = int(n_val / args.batch_size) 
    for i_batch in range(n_val_batches):
        batch_data = val_scenes[i_batch * args.batch_size : 
                                  (i_batch + 1) * args.batch_size]
        alt_indices = \
                [np.random.choice(n_val, size=args.batch_size)
                 for i_alt in range(args.alternatives)]
        alt_data = [[val_scenes[i] for i in alt] for alt in alt_indices]

        outputs = model(batch_data, alt_data)
        targets = target_func(args, batch_data)

        if torch.cuda.is_available():
            targets = targets.cuda()

        if model.name =='Listener0':
            _, predicted = outputs.max(dim=1)
            n_correct = (predicted.data == targets.data).sum()
            total_correct += n_correct

        loss = criterion(outputs, targets)
        epoch_loss += loss.data[0]

    logging.info('====> Epoch %d: Validation loss: %.4f' % (epoch, epoch_loss))
    if model.name=='Listener0':
        logging.info('Validation Accuracy: %f' % (total_correct / (n_val_batches * args.batch_size)))


def train(train_scenes, val_scenes, model, optimizer, args, target_func):
    logging.info('Training %s...' % (model.name))
    n_train = len(train_scenes) 
    model.train()

    criterion = nn.CrossEntropyLoss()

    n_train_batches = int(n_train / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        total_correct = 0.0

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

            if model.name =='Listener0':
                _, predicted = outputs.max(dim=1)
                n_correct = (predicted.data == targets.data).sum()
                total_correct += n_correct

            loss = criterion(outputs, targets) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

            if (i_batch % args.log_interval == 0):
                logging.info('Epoch [%d/%d], Step[%d/%d], loss: %.4f' 
                  %(epoch, args.epochs, i_batch, n_train_batches, loss.data[0]))

        logging.info('====> Epoch %d: Training loss: %.4f' % (epoch, epoch_loss))

        if model.name=='Listener0':
            logging.info('Training Accuracy: %f' % (total_correct / (n_train_batches * args.batch_size)))

        validate(val_scenes, model, optimizer, args, target_func, epoch)


def get_examples(model, train_scenes, args, word_index):
    model.eval()
    n_train = len(train_scenes) 
    n_train_batches = int(n_train / args.batch_size)

    bleu_score = 0

    for i_batch in range(n_train_batches):
        batch_data = train_scenes[i_batch * args.batch_size : 
                                      (i_batch + 1) * args.batch_size]
        alt_indices = [np.random.choice(n_train, size=args.batch_size) for i_alt in range(args.alternatives)]
        alt_data = [[train_scenes[i] for i in alt] for alt in alt_indices]

        probs, sentences = model.sample(batch_data, alt_data, k=10) # [batch_size, sentences] sent is [100 x 10 x 20]
        # print_datas_and_desc(batch_data, alt_data, sentences.data, word_index)
        print_tensor3d(sentences.data, word_index)
        logging.info([(i, (scene.image_id, alt_scene.image_id)) for i, (scene, alt_scene) in enumerate(zip(batch_data, alt_data[0]))])

        save_image_pairs(sentences.squeeze(), batch_data, alt_data, word_index)

        # scores = calculate_bleu(batch_data, sentences.squeeze())
        # for _, score in scores:
            # bleu_score += score
        # logging.info('Current BLEU Score: %f' % (bleu_score / ((i_batch+1) * args.batch_size)))

    bleu_score /= n_train
    return bleu_score

def save_image_pairs(sentences, data, alt_data, WORD_INDEX):
    new_dir = 'pairs/ss1' + str(experiment_counter)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for i, (scene, alt_scene) in enumerate(zip(data, alt_data[0])):
        s = sentences[i] # this is the sentence
        s_joined = tensor_to_caption(s, WORD_INDEX)

        img1 = data_path + 'RenderedScenes/Scene' + str(scene.image_id) + '.png'
        img2 = data_path + 'RenderedScenes/Scene' + str(alt_scene.image_id) + '.png'
        combined_img = new_dir + '/ss1' + str(experiment_counter) + '_' + str(scene.image_id) + '_&_' + str(alt_scene.image_id) + '.png'

        images = map(Image.open, [img1, img2])
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths) + 10
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height + 16))

        # text to images
        d = ImageDraw.Draw(new_im)
        text_width, text_height = d.textsize(s_joined)
        d.text(((total_width - text_width)/2, max_height+2), s_joined, fill=(255, 255, 255))

        x_offset = 0
        images = map(Image.open, [img1, img2])
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0] + 10

        new_im.save(combined_img)

def run_experiment(name, cname, rname, models, data, WORD_INDEX, args):
    data_by_image = defaultdict(list)
    for datum in data:
        data_by_image[datum.image_id].append(datum)

    with open("experiments/%s/%s.ids.txt" % (name, cname)) as id_f, \
        open("experiments/%s/%s.results.%s.txt" % (name, cname, rname), 'w') as results_f:
        results_f.write("id,target,distractor,similarity,model_name,speaker_score,listener_score,description\n")

        counter = 0
        for line in id_f:
            img1, img2, similarity = line.strip().split(',')
            assert img1 in data_by_image and img2 in data_by_image
            d1 = data_by_image[img1][0]
            d2 = data_by_image[img2][0]
            for model_name, model in models.items():

                if model_name == 'speaker0':
                    speaker_scores, all_samples = model.sample([d1], [[d2]], viterbi=False, k=args.k)

                    best_sentence = all_samples.squeeze(0)[0]
                    sentence = tensor_to_caption(best_sentence, WORD_INDEX)
                    save_image_pairs(best_sentence.unsqueeze(0), [d1], [[d2]], WORD_INDEX)

                else:
                    (listener_scores, speaker_scores), samples = \
                            model.sample([d1], [[d2]], viterbi=False, k=args.k)
                    samples = samples.squeeze(0).squeeze(0)
                    sentence = tensor_to_caption(samples, WORD_INDEX)
                    save_image_pairs(samples.unsqueeze(0), [d1], [[d2]], WORD_INDEX)

                parts = [
                    counter,
                    img1,
                    img2,
                    similarity,
                    model_name,
                    speaker_scores.squeeze(0)[0],
                    sentence
                ]

                results_f.write(",".join([str(s) for s in parts]))
                results_f.write('\n')
                counter += 1

def calculate_bleu(batch, candidates):
    scene_to_description = {}
    for scene in batch:
        if scene.image_id in scene_to_description:
            scene_to_description[scene.image_id].append(scene.description)
        else:
            scene_to_description[scene.image_id] = [scene.description]

    candidates_with_ids = [(batch[i].image_id, candidate) for i, candidate in enumerate(candidates)]
    ids_with_scores = []

    for (img_id, candidate) in candidates_with_ids:
        index_of_end = (candidate.data == 2).nonzero()[0][0] if len((candidate.data == 2).nonzero()) > 0 else MAX_LEN
        candidate_chopped = candidate.data[1:index_of_end] if index_of_end > 1 else []
        score = sentence_bleu(scene_to_description[img_id], candidate_chopped)
        ids_with_scores.append((img_id, score))
    return ids_with_scores

def save_model(model, args):
    file_name = 'models/' + args.model + experiment_counter + '.pth'
    torch.save(model.state_dict(), file_name)
    logging.info('Saved model to %s' % (file_name))

def load_model(model, path):
    full_path = 'models/' + path
    logging.info('Loading saved model %s into %s ...' % (path, model.name))

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(full_path, map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load(full_path, map_location=lambda storage, loc: storage))
        model.cpu()

    logging.info('Model loaded.')

def convert_model(model, new_path):
    full_new_path = 'models/' + new_path

def setup_logging(args):
    global experiment_counter
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
