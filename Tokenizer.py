"""
Tokenization class for neural-complexity meant to copy (very loosely) certain functions of HuggingFace's tokenizers library.
"""
import torch
import re

from typing import List, Union, Optional

from data import SentenceCorpus
from data import sent_tokenize

class Tokenizer(SentenceCorpus):

    def __init__(self, vocab_file):
        super().__init__('.', vocab_file, interact_flag=True)
        self.model_max_length = int(1e30)
        self._unk_token = "<unk>"
        self._eos_token = "<eos>"
        self._pad_token = None

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    @property
    def unk_token(self) -> str:
        return str(self._unk_token)

    @property 
    def eos_token(self) -> str:
        return str(self._eos_token)

    @property
    def pad_token(self) -> str:
        return str(self._pad_token)

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the padding token in the vocabulary. Returns :obj:`None` if the token has not been
        set.
        """
        if self._pad_token is None:
            return None
        return self.dictionary.word2idx[self.pad_token]

    @property
    def unk_token_id(self):
        if self.unk_token in self.dictionary.word2idx:
            return self.dictionary.word2idx[self.unk_token]
        else:
            return None

    @property
    def eos_token_id(self):
        if self.eos_token in self.dictionary.word2idx:
            return self.dictionary.word2idx[self.eos_token]
        else:
            return None

    def __len__(self):
        return self.vocab_size

    def __call__(self, line, return_tensors=None):

        encoded = self.encode(line)
        if return_tensors=='pt':
            return {'input_ids': torch.tensor(encoded, dtype=torch.int64).unsqueeze(0), 
                    'attention_mask': torch.tensor([1]*len(encoded)).unsqueeze(0)}
        elif return_tensors is None:
            return {'input_ids': encoded, 
                    'attention_mask': [1]*len(encoded)}
        else:
            sys.stderr.write('I have not implemented a return_tensors type: '+str(return_tensors)+'\n')
            sys.exit(1)

    def batch_encode_plus(self, batch_text: List[str], 
            padding = False,
            return_tensors = None):

        return_inputs = {'input_ids': [], 'attention_mask': []}
        for text in batch_text:
            encoded = self.__call__(text, return_tensors=None)
            return_inputs['input_ids'].append(encoded['input_ids'])
            return_inputs['attention_mask'].append(encoded['attention_mask'])

        if padding:
            assert self.pad_token_id is not None, 'Attempting to PAD with no token'
            max_seq_len = max(len(input_ids) for input_ids in return_inputs['input_ids'])
            padded_batch_outputs = {'input_ids': [], 'attention_mask': []}
            for i in range(len(return_inputs['input_ids'])):
                inputs = return_inputs['input_ids'][i]
                attn = return_inputs['attention_mask'][i]
                inputs = {'input_ids':inputs, 'attention_mask': attn}

                outputs = self._pad(inputs, max_seq_len)

                padded_batch_outputs['input_ids'].append(outputs['input_ids'])
                padded_batch_outputs['attention_mask'].append(outputs['attention_mask'])

            return_inputs = padded_batch_outputs

        if return_tensors == 'pt':
            return_inputs['input_ids'] = torch.tensor(return_inputs['input_ids'], dtype=torch.int64)
            return_inputs['attention_mask'] = torch.tensor(return_inputs['attention_mask'])
            return return_inputs

        elif return_tensors is None:
            return return_inputs

        else:
            sys.stderr.write('I have not implemented a return_tensors type: '+str(return_tensors)+'\n')
            sys.exit(1)

    def _pad(self, batch_element, max_seq_len):

        #needs to be padded
        if len(batch_element['input_ids']) != max_seq_len:
            difference = max_seq_len - len(batch_element['input_ids'])
            batch_element['input_ids'] = batch_element['input_ids'] + [self.pad_token_id]*difference
            batch_element['attention_mask'] = batch_element['attention_mask']+[0]*difference
        return batch_element

    def encode(self, line, add_space_before_punct_symbol=True, lower=True,
            remove_trailing_spaces=True):

        if lower:
            line = line.lower()

        if remove_trailing_spaces:
            line = line.strip()

        if add_space_before_punct_symbol:
            punct = "!\"#$%&'()*+,./:;-<=>?@[\]^_`{|}~"
            #add space before punct
            line = line.translate(str.maketrans({key: " {0}".format(key) for key in punct}))

            #break things like "farm-house" into "farm - house" and "and/or" into "and / or" careful here
            punct = "/-"
            #add space before punct
            line = line.translate(str.maketrans({key: "{0} ".format(key) for key in punct}))

            #remove double spaces
            line = re.sub('\s{2,}', ' ', line)

        sentences = sent_tokenize(line)
        output = []
        for x, sent in enumerate(sentences):
            sent = sent.split(' ')
            if x == 0:
                sent = ['<eos>'] + sent

            #imagine we add a word is this really a sentence
            #If it's a sentence then sent_tokenize will 
            #generate two sentences
            #A bit hacky but it helps in parity with huggingface
            test_sent = ' '.join(sent + ['the'])
            if len(sent_tokenize(test_sent)) != 1:
                sent = sent + ['<eos>']

            output += list(self.convert_to_ids(sent).data.numpy())
        return output

    def convert_ids_to_tokens(self, ids):
        if type(ids) != list:
            ids = [ids]
        return self.decode(ids)

    def decode(self, ids):
        words = list(map(lambda x: self.dictionary.idx2word[x], ids))
        return words

    def convert_tokens_to_ids(self, 
            tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token):
        if token in self.dictionary.word2idx:
            return self.dictionary.word2idx[token]
        return self.unk_token_id
