import random

from torch.utils.data import Dataset
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class config:
    n = 16518
    m = 11794

def collate_batch(it, st, end):
    MAX_LENGTH = 64
    text_list = []
    u_list = []
    pos_list = []
    neg_list = []
    r_list = []

    for i in range(st, end):
        (u, v, k, r, _text) = it[i]
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        if processed_text.shape[0] > MAX_LENGTH:
            processed_text = processed_text[:MAX_LENGTH]
        else:
            processed_text = torch.cat([processed_text, torch.zeros(MAX_LENGTH - processed_text.size()[0],
                                                                    dtype=torch.int64)], dim=0)
        assert processed_text.size()[0] == MAX_LENGTH
        assert processed_text.dtype == torch.int64
        text_list.append(processed_text)
        u_list.append(u)
        pos_list.append(v)
        neg_list.append(k)
        r_list.append(r)

    u_list = torch.tensor(u_list, dtype=torch.int64)
    pos_list = torch.tensor(pos_list, dtype=torch.int64)
    neg_list = torch.tensor(neg_list, dtype=torch.int64)
    r_list = torch.tensor(r_list)
    text_list = pad_sequence(text_list)  # T,B
    return u_list, pos_list, neg_list, r_list, text_list

class AmazonMusic(Dataset):
    def __init__(self, m):
      with open('texts.pkl', 'rb') as f:
        self.data = pickle.load(f)
        self.data = self.data
        self.m = m
        print(f'loading data, len: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], random.choice(range(self.m)), self.data[idx][2], self.data[idx][3]

tokenizer = get_tokenizer('basic_english')
train = AmazonMusic(config.m)
def yield_tokens(data_iter):
    for _,__,___,____, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))