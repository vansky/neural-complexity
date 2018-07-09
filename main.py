from __future__ import print_function
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import data
import model
import sys
import numpy as np
import warnings

try:
    from progress.bar import Bar
    PROGRESS = True
except:
    PROGRESS = False

warnings.filterwarnings("ignore") #suppress SourceChangeWarnings

sys.stderr.write('Libraries loaded\n')

## Parallelization notes:
##   Does not currently operate across multiple nodes
##   Single GPU is better for default: tied,emsize:200,nhid:200,nlayers:2,dropout:0.2
##
##   Multiple GPUs are better for tied,emsize:1500,nhid:1500,nlayers:2,dropout:0.65
##      4 GPUs train on wikitext-2 in 1/2 - 2/3 the time of 1 GPU

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')

## Model parameters
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--cache', action='store_true',
                    help='add Grave et al., (2016) neural cache to pre-trained model (training again will tune cache on dev)')

## Data parameters
parser.add_argument('--model_file', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--cache_model_file', type=str,  default='cache_model.pt',
                    help='path to save the tuned cache model')
parser.add_argument('--data_dir', type=str, default='./data/wikitext-2',
                    help='location of the corpus data')
parser.add_argument('--vocab_file', type=str, default='vocab.txt',
                    help='path to save the vocab file')
parser.add_argument('--trainfname', type=str, default='train.txt',
                    help='name of the training file')
parser.add_argument('--validfname', type=str, default='valid.txt',
                    help='name of the validation file')
parser.add_argument('--testfname', type=str, default='test.txt',
                    help='name of the test file')

## Runtime parameters
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--single', action='store_true',
                    help='use only a single GPU (even if more are available)')
parser.add_argument('--test', action='store_true',
                    help='test a trained LM')
parser.add_argument('--words', action='store_true',
                    help='evaluate word-level complexities (instead of sentence-level loss)')
parser.add_argument('--interact', action='store_true',
                    help='run a trained network interactively')
parser.add_argument('--csep', type=str, default=' ',
                    help='change the separator in the complexity output')
parser.add_argument('--guess', action='store_true',
                    help='display best guesses at each time step')
parser.add_argument('--guessn', type=int, default=1,
                    help='output top n guesses')
parser.add_argument('--guessscores', action='store_true',
                    help='display guess scores along with guesses')
parser.add_argument('--guessratios', action='store_true',
                    help='display guess ratios normalized by best guess')
parser.add_argument('--guessprobs', action='store_true',
                    help='display guess probs along with guesses')
parser.add_argument('--complexn', type=int, default=0,
                    help='compute complexity only over top n guesses (0 = all guesses)')
parser.add_argument('--softcliptopk', action="store_true",
                    help='soften non top-k options instead of removing them')
parser.add_argument('--nopp', action='store_true',
                    help='suppress evaluation perplexity output')

parser.add_argument('--reset_cache', action='store_true',
                    help='reset cache but retain hyperparameter settings')
parser.add_argument('--manual_cache_tuning', action='store_true',
                    help='force manual cache tuning from runtime parameters even if cache was previously tuned')
parser.add_argument('--cache_hidden_type', type=str, default='flat',
                    help='type of hidden layer used for cache {"top","bottom","flat"}')
parser.add_argument('--cache_size', type=int, default=2000,
                    help='size of neural cache (in time steps)')
parser.add_argument('--cache_lambda', type=float, default=0.1,
                    help='weight of cache vs training (0.0 = all training; 1.0 = all cache) ')
parser.add_argument('--cache_theta', type=float, default=0.3,
                    help='multiplied by current:cache similarity in cache score (0.0 = no cache similarity) ')

args = parser.parse_args()

if args.interact:
    # If in interactive mode, force complexity output
    args.words = True
    args.test = True

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
#  a g m s
#  b h n t
#  c i o u
#  d j p v
#  e k q w
#  f l r x
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # Turning the data over to CUDA at this point may lead to more OOM errors
    #if args.cuda:
     #    data = data.cuda()
    return data

eval_batch_size = 10

corpus = data.SentenceCorpus(args.data_dir, args.vocab_file, args.test, args.interact,
                             trainfname=args.trainfname,
                             validfname=args.validfname,
                             testfname=args.testfname)

if not args.interact:
    if args.test:
        test_sents, test_data = corpus.test
    else:
        train_data = batchify(corpus.train, args.batch_size)
        val_data = batchify(corpus.valid, eval_batch_size)

###############################################################################
# Build/load the model
###############################################################################

if not args.test and not args.interact:
    ntokens = len(corpus.dictionary)
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    if args.cuda:
        if (not args.single) and (torch.cuda.device_count() > 1):
            # Scatters minibatches (in dim=1) across available GPUs
            model = nn.DataParallel(model,dim=1)
        model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Complexity measures
###############################################################################

def get_entropy(o):
    ## o should be a vector scoring possible classes
    ## returns a scalar entropy over o
    if args.complexn == 0:
        beam = o
    else:
        # duplicate o but with all losing guesses set to 0
        beamk,beamix = torch.topk(o,args.complexn,0)
        if args.cuda:
            if args.softcliptopk:
                beam = Variable(torch.cuda.FloatTensor(o.size()).fill_(0)).scatter(0,beamix,beamk)
            else:
                beam = Variable(torch.cuda.FloatTensor(o.size()).fill_(float("-inf"))).scatter(0,beamix,beamk)
        else:
            if args.softcliptopk:
                beam = Variable(torch.zeros(o.size())).scatter(0,beamix,beamk)
            else:
                beam = Variable(torch.FloatTensor(o.size()).fill_(float("-inf"))).scatter(0,beamix,beamk)

    probs = nn.functional.softmax(beam,dim=0)
    logprobs = nn.functional.log_softmax(beam,dim=0) #numerically more stable than two separate operations
    prod = probs * logprobs
    prod[prod != prod] = 0 #set nans to 0
    return -1 * torch.sum(prod)

def get_surps(o):
    ## o should be a vector scoring possible classes
    ## returns a vector containing the surprisal of each class in o
    if args.complexn == 0:
        beam = o
    else:
        # duplicate o but with all losing guesses set to 0
        beamk,beamix = torch.topk(o,args.complexn,0)
        if args.cuda:
            if args.softcliptopk:
                beam = Variable(torch.cuda.FloatTensor(o.size()).fill_(0)).scatter(0,beamix,beamk)
            else:
                beam = Variable(torch.cuda.FloatTensor(o.size()).fill_(float("-inf"))).scatter(0,beamix,beamk)
        else:
            if args.softcliptopk:
                beam = Variable(torch.zeros(o.size())).scatter(0,beamix,beamk)
            else:
                beam = Variable(torch.FloatTensor(o.size()).fill_(float("-inf"))).scatter(0,beamix,beamk)

    logprobs = nn.functional.log_softmax(beam,dim=0)
    return -1 * logprobs

def get_guesses(o,scores=False):
    ## o should be a vector scoring possible classes
    guessvals, guessixes = torch.topk(o,args.guessn,0)
    # guessvals are the scores of each input cell
    # guessixes are the indices of the max cells
    if scores:
        return guessvals
    else:
        return guessixes

def get_guessscores(o):
    return get_guesses(o,True)

def get_complexity(o,t,sentid):
    Hs = torch.squeeze(apply(get_entropy,o))
    surps = apply(get_surps,o)

    if args.guess:
        guesses = apply(get_guesses, o)
        guessscores = apply(get_guessscores, o)
    ## Use dimensional indexing method
    ## NOTE: For some reason, this doesn't work.
    ##       May marginally speed things if we can determine why
    ##       Currently 'probs' ends up equivalent to o after the softmax
    #probs = nn.functional.softmax(o,dim=0)
    #logprobs = nn.functional.log_softmax(o,dim=0)
    #Hs = -1 * torch.sum(probs * logprobs),dim=1)
    #surps = -1 * logprobs
    ## Move along
    for corpuspos,targ in enumerate(t):
        word = corpus.dictionary.idx2word[int(targ)]
        if word == '<eos>':
            ## don't output the complexity of EOS
            continue
        surp = surps[corpuspos][int(targ)]
        if args.guess:
            outputguesses = []
            for g in range(args.guessn):
                outputguesses.append(corpus.dictionary.idx2word[int(guesses[corpuspos][g])])
                if args.guessscores:
                    ## output raw scores
                    outputguesses.append("{:.3f}".format(float(guessscores[corpuspos][g])))
                elif args.guessratios:
                    ## output scores (ratio of score(x)/score(best guess)
                    outputguesses.append("{:.3f}".format(float(guessscores[corpuspos][g])/float(guessscores[corpuspos][0])))
                elif args.guessprobs:
                  ##output probabilities ## Currently normalizes probs over N-best list; ideally it'd normalize to probs before getting the N-best
                  outputguesses.append("{:.3f}".format(math.exp(float(nn.functional.log_softmax(guessscores[corpuspos],dim=0)[g]))))
            outputguesses = args.csep.join(outputguesses)
            print(args.csep.join([str(word),str(sentid),str(corpuspos),str(len(word)),str(float(surp)),str(float(Hs[corpuspos])),str(max(0,float(Hs[max(corpuspos-1,0)])-float(Hs[corpuspos]))),str(outputguesses)]))
        else:
            print(args.csep.join([str(word),str(sentid),str(corpuspos),str(len(word)),str(float(surp)),str(float(Hs[corpuspos])),str(max(0,float(Hs[max(corpuspos-1,0)])-float(Hs[corpuspos])))]))

def apply(func, M):
    ## applies a function along a given dimension
    tList = [func(m) for m in torch.unbind(M, dim=0) ]
    res = torch.stack(tList, dim=0)
    return res

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
#  a g m s      b h n t
#  b h n t      c i o u
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def test_get_batch(source, evaluation=False):
    seq_len = len(source) - 1
    data = Variable(source[:seq_len], volatile=evaluation)
    target = Variable(source[1:1+seq_len].view(-1))
    # This is where data should be CUDA-fied to lessen OOM errors
    if args.cuda:
        return data.cuda(), target.cuda()
    else:
        return data, target

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    #This is where data should be CUDA-fied to lessen OOM errors
    if args.cuda:
        return data.cuda(), target.cuda()
    else:
        return data, target

def test_evaluate(test_sentences, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    if args.complexn > ntokens or args.complexn <= 0:
        args.complexn = ntokens
        sys.stderr.write('Using beamsize: '+str(ntokens)+'\n')
    else:
        sys.stderr.write('Using beamsize: '+str(args.complexn)+'\n')

    if args.words:
        if args.complexn == ntokens:
            print('word{0}sentid{0}sentpos{0}wlen{0}surp{0}entropy{0}entred'.format(args.csep), end='')
        else:
            print('word{0}sentid{0}sentpos{0}wlen{0}surp{1}{0}entropy{1}{0}entred{1}'.format(args.csep,args.complexn), end='')
        if args.guess:
            for i in range(args.guessn):
                print('{0}guess'.format(args.csep)+str(i), end='')
                if args.guessscores:
                    print('{0}gscore'.format(args.csep)+str(i), end='')
                elif args.guessprobs:
                    print('{0}gprob'.format(args.csep)+str(i), end='')
                elif args.guessratios:
                    print('{0}gratio'.format(args.csep)+str(i), end='')
        sys.stdout.write('\n')
    if PROGRESS:
        bar = Bar('Processing', max=len(data_source))
    for i in range(len(data_source)):
        sent_ids = data_source[i]
        sent = test_sentences[i]
        if args.cuda:
            sent_ids = sent_ids.cuda()
        if args.cuda and (not args.single) and (torch.cuda.device_count() > 1):
            # "module" is necessary when using DataParallel
            hidden = model.module.init_hidden(1) # number of parallel sentences being processed
        else:
            hidden = model.init_hidden(1) # number of parallel sentences being processed
        data, targets = test_get_batch(sent_ids, evaluation=True)
        data=data.unsqueeze(1) # only needed if there is just a single sentence being processed
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        curr_loss = criterion(output_flat, targets).data
        #curr_loss = len(data) * criterion(output_flat, targets).data # needed if there is more than a single sentence being processed
        total_loss += curr_loss
        if args.words:
            # output word-level complexity metrics
            get_complexity(output_flat,targets,i)
        else:
            # output sentence-level loss
            print(str(sent)+":"+str(curr_loss[0]))
        hidden = repackage_hidden(hidden)
        if PROGRESS:
            bar.next()
    if PROGRESS:
        bar.finish()
    return total_loss[0] / len(data_source)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    if args.cuda and (not args.single) and (torch.cuda.device_count() > 1):
        #"module" is necessary when using DataParallel
        hidden = model.module.init_hidden(eval_batch_size)
    else:
        hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        curr_loss = len(data) * criterion(output_flat, targets).data
        total_loss += curr_loss
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.cuda and (not args.single) and (torch.cuda.device_count() > 1):
        # "module" is necessary when using DataParallel
        hidden = model.module.init_hidden(args.batch_size)
    else:
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
prev_val_loss = None
prev2_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
if not args.test:
    if not args.cache:
        ## Train model
        try:
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train()
                val_loss = evaluate(val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.model_file, 'wb') as f:
                        torch.save(model, f)
                        best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    if val_loss == prev_val_loss == prev2_val_loss:
                        print('Covergence achieved! Ending training early')
                        break
                    prev2_val_loss = prev_val_loss
                    prev_val_loss = val_loss
                    lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    else:
        # Load the best saved model.
        with open(args.model_file, 'rb') as f:
            if args.cuda:
                #Run on GPUs
                model = torch.load(f)
            else:
                #Run on CPUs
                model = torch.load(f, map_location=lambda storage, loc: storage)

        model.init_cache(cache_size=args.cache_size,batch_size=eval_batch_size,cache_hidden_type=args.cache_hidden_type)
        ## Tune cache hyperparameters
        for l in np.arange(0.05,0.55,0.1):
            for theta in np.arange(0.1,1.1,0.1):
                model.cache_theta = theta
                model.cache_lambda = l
                val_loss = evaluate(val_data)
                print('-' * 89)
                print('| end of lambda {:3.2f} | end of theta {:3.2f} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(l, theta, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.cache_model_file, 'wb') as f:
                        torch.save(model, f)
                        best_val_loss = val_loss
                model.reset_cache(batch_size=eval_batch_size)
else:
    # Load the best saved model.
    with open(args.model_file, 'rb') as f:
        if args.cuda:
            #Run on GPUs
            model = torch.load(f)
        else:
            #Run on CPUs
            model = torch.load(f, map_location=lambda storage, loc: storage)
    if not hasattr(model,'use_cache'):
        ## Backwards compatibility for models trained without a cache
        model.use_cache = False
    if args.cache and (not model.use_cache or args.manual_cache_tuning):
        ## Use an default tuned or manually tuned cache
        ## To tune the cache, use --cache on a pretrained model without --test
        model.init_cache(cache_size=args.cache_size,batch_size=1,cache_theta=args.cache_theta,cache_lambda=args.cache_lambda)
        with open(args.cache_model_file, 'wb') as f:
            torch.save(model, f)

    if args.cache:
        if args.reset_cache:
            model.reset_cache(batch_size=1)
        elif model.hidden_cache.shape[-1] != 1:
            print('Warning: Testing must be done with a batch_size of 1.\n\
                   Forcing a cache_reset/reshape to achieve a batch_size of 1.\n\
                   In future, either use the --reset_cache option during testing\n\
                   or use the "--batch_size 1" option during training.')
            model.reset_cache(batch_size=1)
        else:
            print('Warning: The cache was not reset after training,\n\
                   so the initial test cache is now filled with the end of the training data.')

    # Run on test data.
    if args.interact:
        # First fix Python 2.x input command
        try: input = raw_input
        except NameError: pass

        # Then run interactively
        print('Running in interactive mode. Ctrl+c to exit')
        try:
           while True:
               instr = input('Input a sentence: ')
               test_sents, test_data = corpus.online_tokenize_with_unks(instr)
               test_evaluate(test_sents, test_data)
        except KeyboardInterrupt:
            print(' ')
            pass
    else:
        test_loss = test_evaluate(test_sents, test_data)
    if not args.interact and not args.nopp:
        print('=' * 89)
        print('| End of testing | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
