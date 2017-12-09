import os
import torch
import dill
from nltk import sent_tokenize

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2idx["<unk>"] = 0
        self.idx2word.append("<unk>")

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class SentenceCorpus(object):
    def __init__(self, path, save_to, testflag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt'):
        if not testflag:
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, trainfname))
            self.valid = self.tokenize(os.path.join(path, validfname))
            self.save_to = self.save_dict(os.path.join(path, save_to))
        else:
            self.dictionary = self.load_dict(os.path.join(path, save_to))
            self.test = self.tokenize_with_unks(os.path.join(path, testfname))

    def save_dict(self, path):
        with open(path, 'wb') as f:
            torch.save(self.dictionary, f, pickle_module=dill)

    def load_dict(self, path):
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            return(torch.load(f, pickle_module=dill))
            
    def tokenize_with_unks(self, path):
        """Tokenizes a text file and replaces unks."""
        assert os.path.exists(path)
        all_ids = []
        sents = []
        with open(path, 'r') as f:
            for fchunk in f:
                for line in sent_tokenize(fchunk):
                  sents.append(line.strip())
                  words = line.split() + ['<eos>']
                  tokens = len(words)

                  # tokenize file content
                  ids = torch.LongTensor(tokens)
                  token = 0
                  for word in words:
                      if word not in self.dictionary.word2idx:
                          ids[token] = self.dictionary.word2idx["<unk>"]
                      else:
                          ids[token] = self.dictionary.word2idx[word]
                      token += 1
                  all_ids.append(ids)
        return (sents, all_ids)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        all_ids = []
        sents = []
        # Add words to the dictionary
        with open(path, 'r') as f:
            for fchunk in f:
                for line in sent_tokenize(fchunk):
                    words = line.strip().split() + ['<eos>']
                    if len(words) == 1:
                        # This was a blank line, so skip it
                        continue
                    sents.append(words[:-1])
                    all_ids.append(torch.LongTensor(len(words)))
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        for sentix,sent in enumerate(sents):
#        with open(path, 'r') as f:
#            for lineix,line in enumerate(f):
            token = 0
#            words = line.split() + ['<eos>']
            for word in sent:
                all_ids[sentix][token] = self.dictionary.word2idx[word]
                token += 1
        return (sents, all_ids)
        
class Corpus(object):
    def __init__(self, path, save_to):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize_with_unks(os.path.join(path, 'test.txt'))
        self.save_to = self.save(os.path.join(path, save_to))
        
    def save(self, path):
        #assert os.path.exists(path)
        with open(path, 'wb') as f:
            torch.save((self.train, self.valid, self.test, self.dictionary), f, pickle_module=dill)
        
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def tokenize_with_unks(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        all_ids = []
        with open(path, 'r') as f:
            #tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens = len(words)

                # tokenize file content
                ids = torch.LongTensor(tokens)
                token = 0
                for word in words:
                    if word not in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx["<unk>"]
                    else:
                        ids[token] = self.dictionary.word2idx[word]
                    token += 1
                all_ids.append(ids)
        
        return all_ids
