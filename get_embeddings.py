'''
Code to extract all the vocabulary embeddings from a neural language model.
'''

from __future__ import print_function
import argparse
import time
import math
import sys
import warnings
import torch
import torch.nn as nn
import data
import model

try:
    from progress.bar import Bar
    PROGRESS = True
except ModuleNotFoundError:
    PROGRESS = False

# suppress SourceChangeWarnings
warnings.filterwarnings("ignore")

sys.stderr.write('Libraries loaded\n')

# Parallelization notes:
#   Does not currently operate across multiple nodes
#   Single GPU is better for default: tied,emsize:200,nhid:200,nlayers:2,dropout:0.2
#
#   Multiple GPUs are better for tied,emsize:1500,nhid:1500,nlayers:2,dropout:0.65
#      4 GPUs train on wikitext-2 in 1/2 - 2/3 the time of 1 GPU

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')

# Model parameters
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

# Data parameters
parser.add_argument('--model_file', type=str, default='model.pt',
                    help='path to model')

parser.add_argument('--vocab_file', type=str, default='vocab.txt',
                    help='path to vocab_file')

args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        sys.stderr.write("WARNING: You have a CUDA device, so you should probably run with --cuda\n")
    else:
        #torch.cuda.manual_seed(args.seed)
        if torch.cuda.device_count() == 1:
            args.single = True

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load the vocab
###############################################################################
idx2word = {}
with open(args.vocab_file, 'r') as f:
    idx = 0
    for line in f:
        line = line.strip()
        idx2word[idx] = line
        idx += 1

###############################################################################
# Load the model
###############################################################################

with open(args.model_file, 'rb') as f:
    if args.cuda:
        model = torch.load(f).to(device)
    else:
        model = torch.load(f, map_location='cpu')

    if args.cuda and (not args.single) and (torch.cuda.device_count() > 1):
        # If applicable, use multi-gpu for training
        # Scatters minibatches (in dim=1) across available GPUs
        model = nn.DataParallel(model, dim=1)
    if isinstance(model, torch.nn.DataParallel):
        # if multi-gpu, access real model for training
        model = model.module
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

#print(model.encoder(torch.LongTensor([w for w in range(model.encoder.num_embeddings)])))
for idx, embed in enumerate(model.encoder(torch.LongTensor([w for w in range(model.encoder.num_embeddings)])).data.numpy().tolist()):
    word = idx2word[idx]

    print(word+' '+' '.join(str(f) for f in embed))
