import argparse
import math
import numpy as np
import random
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model, utils, corpus

parser = argparse.ArgumentParser(description='Pragmatics')
# parser.add_argument('--model', help='which model to use')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argumet('--hidden-sz', type=int, default=100,
                    help='hidden size of the neural networks')
args = parser.parse_args()

torch.manual_seed(args.seed)


WORD_EMBEDDING_SZ = 50
PROP_EMBEDDING_SZ = 50

RHO = 0.95
EPS = 0.000001
LR = 1
CLIP = 10

N_TEST_IMAGES = 100
N_TEST = N_TEST_IMAGES * 10
N_EXPERIMENT_PAIRS = 100

train_scenes, dev_scenes, test_scenes = corpus.load_abstract()

listener0_model = model.Listener0Model(args.hidden_sz) # need to pass in some parameters
speaker0_model = model.Speaker0Model(args.hidden_sz)
sampling_speaker1_model = model.SamplingSpeaker1Model(args.hidden_sz)
# compiled_speaker1_model = model.Compiledspeaker1Model()

# Train base
utils.train(train_scenes, dev_scenes, listener0_model)
utils.train(train_scenes, dev_scenes, speaker0_model)

# Train compiled
utils.train(train_scenes, dev_scenes, sampling_speaker1_model)

