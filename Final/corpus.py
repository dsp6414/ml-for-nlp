from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import re
import pdb
import os

from util import Index

WORD_INDEX = Index()

Prop = namedtuple("Prop", ["type_index", "object_index", "x", "y", "z", "flip"])
Scene = namedtuple("Scene", ["image_id", "props", "description", "features"])

N_IMAGES = 10020
N_DEV_IMAGES = 1000
N_TEST_IMAGES = 1000
MIN_WORD_COUNT = 5
DEV_RANGE = range(N_IMAGES - N_TEST_IMAGES - N_DEV_IMAGES, N_IMAGES - N_TEST_IMAGES)
TEST_RANGE = range(N_IMAGES - N_TEST_IMAGES, N_IMAGES)

# data_path = '../../../AbstractScenes_v1.1/'
data_path = 'AbstractScenes_v1.1/'

def load_props():
    scene_props = []
    with open(data_path + "Scenes_10020.txt") as scene_f:
        scene_f.readline()
        while True:
            line = scene_f.readline().strip()
            if not line:
                break
            length = int(line.split()[1])
            props = []
            for i_object in range(length):
                line = scene_f.readline().strip()
                parts = line.split()[1:]
                parts = [int(p) for p in parts]
                props.append(Prop(*parts))

            scene_props.append(props)
    return scene_props

# Normalize each feature inside props
def normalize_props(scene_props):
    feats = np.zeros(4)
    feats_sq = np.zeros(4)
    count = 0

    for props in scene_props:
        for prop in props:
            feats_here = np.asarray([prop.x, prop.y, prop.z, prop.flip])
            feats += feats_here
            feats_sq += feats_here ** 2
            count += 1

    mean = feats / count
    std = np.sqrt(feats_sq / count - mean ** 2)
    assert (std > 0).all()

    norm_scene_props = []
    for props in scene_props:
        new_props = []
        for prop in props:
            prop_feats = np.asarray([prop.x, prop.y, prop.z, prop.flip], dtype=float)
            prop_feats -= mean
            prop_feats /= std
            x, y, z, flip = prop_feats
            new_prop = Prop(prop.type_index, prop.object_index, x, y, z, flip)
            new_props.append(new_prop)
        norm_scene_props.append(new_props)
    return norm_scene_props

def load_scenes(scene_props):
    scenes = []

    word_counter = defaultdict(lambda: 0)

    feature_df = load_all_feature_files()
    

    # Read in the sentence descriptions
    for sent_file_id in range(1, 3):
        with open(data_path + "SimpleSentences/SimpleSentences%d_10020.txt" %
                sent_file_id) as sent_f:
            for sent_line in sent_f:
                sent_parts = sent_line.strip().split("\t")
                if len(sent_parts) >= 3: # every other line is blank so you gotta skip
                    sent = sent_parts[2]
                    sent = sent.replace('"', ' " ')
                    sent = sent.replace("'", " ' ")
                    sent = re.sub(r"[.?!]", "", sent)
                    words = sent.lower().split()
                    words = ["<s>"] + words + ["</s>"]
                    for word in words:
                        word_counter[word] += 1
    for word, count in word_counter.items():
        if count >= MIN_WORD_COUNT:
            WORD_INDEX.index(word)

    # Read in the ids
    for sent_file_id in range(1, 3):
        with open(data_path + "SimpleSentences/SimpleSentences%d_10020.txt" %
                sent_file_id) as sent_f:
            for sent_line in sent_f:
                sent_parts = sent_line.strip().split("\t")
                if len(sent_parts) >= 3: # skip empty lines
                    scene_id = int(sent_parts[0])
                    props = scene_props[scene_id]

                    sent_id = int(sent_parts[1])
                    image_id = scene_id / 10
                    image_subid = scene_id % 10
                    image_strid = "%d_%d" % (image_id, image_subid)

                    sent = sent_parts[2]
                    sent = sent.replace('"', "")
                    sent = re.sub(r"[.?!']", "", sent)
                    words = sent.lower().split()
                    words = ["<s>"] + words + ["</s>"]
                    word_ids = [WORD_INDEX[w] or 0 for w in words]

                    print(scene_id)                 
                    features = feature_df.iloc[scene_id, :]
                    scenes.append(Scene(image_strid, props, word_ids, features))
    return scenes

def load_binarized_feature_file(file_path):
    feature_df = pd.read_csv(file_path, sep='\t', header=None)
    feature_df.dropna(axis=1, how='all', inplace=True)
    return feature_df

def load_all_feature_files():
    feature_dfs = []
    for file in os.listdir(data_path + "VisualFeatures"):
        if not file.endswith("_names.txt"):
            file_path = data_path + "VisualFeatures/" + file
            print(file_path)
            feature_df = load_binarized_feature_file(file_path)
            feature_dfs.append(feature_df)

    pdb.set_trace()
    merged_df = pd.concat(feature_dfs, axis=0)
    return merged_df


    


def load_abstract():
    props = load_props()
    norm_props = normalize_props(props)
    scenes = load_scenes(norm_props)
    train_scenes = []
    dev_scenes = []
    test_scenes = []
    for scene in scenes:
        raw_id = int(scene.image_id.replace("_", ""))
        if raw_id in DEV_RANGE:
            dev_scenes.append(scene)
        elif raw_id in TEST_RANGE:
            test_scenes.append(scene)
        else:
            train_scenes.append(scene)
    return train_scenes, dev_scenes, test_scenes