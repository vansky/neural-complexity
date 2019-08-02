"""
Generates sentences by sampling from a language model
"""

import argparse
import torch
import data

parser = argparse.ArgumentParser(description='PyTorch Language Model')

# Model parameters.
parser.add_argument('--data_dir', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model_file', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--numwords', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--sentences', action='store_true',
                    help='generate one sentence per line')
parser.add_argument('--single', action='store_true',
                    help='use only a single GPU (even if more are available)')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log_interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--vocab_file', type=str, default='vocab.txt',
                    help='path to save the LM data')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.model_file, 'rb') as f:
    if args.cuda:
        model = torch.load(f).to(device)
    else:
        model = torch.load(f, map_location='cpu')
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    if args.cuda and (not args.single) and (torch.cuda.device_count() > 1):
        model.module.rnn.flatten_parameters()
    else:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.rnn.flatten_parameters()
model.eval()

corpus = data.SentenceCorpus(args.data_dir, args.vocab_file, generate_flag=True)

ntokens = len(corpus.dictionary)
if args.cuda and (not args.single) and (torch.cuda.device_count() > 1):
    hidden = model.module.init_hidden(1)
else:
    hidden = model.init_hidden(1)
input_sequence = torch.rand(1, 1).mul(ntokens).long()
if args.cuda:
    input_sequence.data = input_sequence.data.to(device)

with open(args.outf, 'w') as outf:
    for i in range(args.numwords):
        output, hidden = model(input_sequence, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input_sequence.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        if args.sentences:
            outf.write(word + ('\n' if word == '<eos>' else ' '))
        else:
            outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.numwords))
    if args.sentences:
        while word != '<eos>':
            output, hidden = model(input_sequence, hidden)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_sequence.data.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            if args.sentences:
                outf.write(word + ('\n' if word == '<eos>' else ' '))
            else:
                outf.write(word + ('\n' if i % 20 == 19 else ' '))
