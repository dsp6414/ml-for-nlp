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

import model, util, corpus

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
parser.add_argument('--hidden-sz', type=int, default=50,
                    help='hidden size of the neural networks')
parser.add_argument('--LR', type=float, default=1.0,
                    help='learning rate')
parser.add_argument('--alternatives', type=int, default=1,
                    help='how many alternatives to find')
args = parser.parse_args()

torch.manual_seed(args.seed)


WORD_EMBEDDING_SZ = 50
PROP_EMBEDDING_SZ = 50

RHO = 0.95
EPS = 0.000001
LR = .1
CLIP = 10

N_TEST_IMAGES = 100
N_TEST = N_TEST_IMAGES * 10
N_EXPERIMENT_PAIRS = 100

VOCAB_SIZE = 2713

# Not sure about num_scenes??
NUM_SCENES = 10

train_scenes, dev_scenes, test_scenes = corpus.load_abstract()

output_size = 1 # should this be 1? because the output of the listener should just be a probability distr.
listener0_model = model.Listener0Model(VOCAB_SIZE, NUM_SCENES, args.hidden_sz, output_size, dropout=0) # need to pass in some parameters
speaker0_model = model.Speaker0Model(VOCAB_SIZE, args.hidden_sz, dropout=0)

# Not sure what output size of speaker model is..
sampling_speaker1_model = model.SamplingSpeaker1Model(VOCAB_SIZE, NUM_SCENES, args.hidden_sz, VOCAB_SIZE, dropout=0)
# compiled_speaker1_model = model.Compiledspeaker1Model()

optimizer_l0 = optim.Adam(listener0_model.parameters(), lr=LR)
optimizer_s0 = optim.Adam(listener0_model.parameters(), lr=LR)
optimizer_ss1 = optim.Adam(listener0_model.parameters(), lr=LR)

# Train base
util.train(train_scenes, dev_scenes, listener0_model, optimizer_l0, args)
util.train(train_scenes, dev_scenes, speaker0_model, optimizer_s0, args)

# Train compiled
util.train(train_scenes, dev_scenes, sampling_speaker1_model, optimizer_ss1, args)

