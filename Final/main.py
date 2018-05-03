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
import logging


import model, util, corpus

parser = argparse.ArgumentParser(description='Pragmatics')
# parser.add_argument('--model', help='which model to use')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-sz', type=int, default=50,
                    help='hidden size of the neural networks')
parser.add_argument('--LR', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--alternatives', type=int, default=1,
                    help='how many alternatives to find')
parser.add_argument('--dropout', type=float, default =0.0, help='dropout probability')
parser.add_argument('--model', default=None, help='which model to train (if debugging)')
parser.add_argument('--dec', default='LSTM', help='which string decoder model to use for Speaker0. LSTM or MLP')

args = parser.parse_args()

torch.manual_seed(args.seed)


WORD_EMBEDDING_SZ = 50
PROP_EMBEDDING_SZ = 50

RHO = 0.95
EPS = 0.000001
LR = args.LR
CLIP = 10

N_TEST_IMAGES = 100
N_TEST = N_TEST_IMAGES * 10
N_EXPERIMENT_PAIRS = 100

# Not sure about num_scenes??
NUM_SCENES = 10

util.setup_logging(args)

train_scenes, dev_scenes, test_scenes = corpus.load_abstract()
VOCAB_SIZE = len(corpus.WORD_INDEX)

VOCAB_SIZE = len(corpus.WORD_INDEX)

output_size = 1 # should this be 1? because the output of the listener should just be a probability distr.
listener0_model = model.Listener0Model(VOCAB_SIZE, NUM_SCENES, args.hidden_sz, output_size, args.dropout) # need to pass in some parameters
speaker0_model = model.Speaker0Model(VOCAB_SIZE, args.hidden_sz, args.dropout, args.dec)

# Not sure what output size of compiled speaker model is..
compiled_speaker1_model = model.CompiledSpeaker1Model(VOCAB_SIZE, NUM_SCENES, args.hidden_sz, output_size, args.dropout)
# compiled_speaker1_model = model.Compiledspeaker1Model()



if torch.cuda.is_available():
	listener0_model.cuda()
	speaker0_model.cuda()
	compiled_speaker1_model.cuda()
	# compiled_speaker1_model.cuda()


optimizer_l0 = optim.Adam(listener0_model.parameters(), lr=LR)
optimizer_s0 = optim.Adam(speaker0_model.parameters(), lr=LR)
optimizer_cs1 = optim.Adam(compiled_speaker1_model.parameters(), lr=LR)

logging.info("Hyperparameters:" + str(args))

if args.model == None:
	logging.info("Listener0: " + str(listener0_model))
	logging.info("Speaker0: " + str(listener0_model))
	logging.info("CompiledSpeaker1Model: " + str(compiled_speaker1_model))
	# Train base
	util.train(train_scenes, listener0_model, optimizer_l0, args, util.listener_targets)
	util.train(train_scenes, speaker0_model, optimizer_s0, args, util.speaker0_targets)

	# Train compiled
	util.train(train_scenes, compiled_speaker1_model, optimizer_cs1, args)
elif args.model == 'l0':
	logging.info("Listener0: " + str(listener0_model))
	util.train(train_scenes, listener0_model, optimizer_l0, args, util.listener_targets)
elif args.model == 's0':
	logging.info("Speaker0: " + str(speaker0_model))
	util.train(train_scenes, speaker0_model, optimizer_s0, args, util.speaker0_targets)
	util.get_examples(speaker0_model, train_scenes, args)
elif args.model == 'cs1':
	logging.info("CompiledSpeaker1Model: " + str(compiled_speaker1_model))
	# util.train(train_scenes, listener0_model, optimizer_l0, args, util.listener_targets)
	# util.train(train_scenes, speaker0_model, optimizer_s0, args, util.speaker0_targets)
	util.train(train_scenes, compiled_speaker1_model, optimizer_cs1, args, util.speaker0_targets)

