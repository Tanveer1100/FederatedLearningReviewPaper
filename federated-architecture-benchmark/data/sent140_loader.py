import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class Sent140Dataset(Dataset):
    def __init__(self, json_path, user, vocab=None, max_len=50):
        with open(json_path, 'r') as f:
            data = json.load(f)['user_data'][user]
        self.raw_x = data['x']
        self.y = data['y']
        self.max_len = max_len

        # Basic tokenizer (word-based)
        self.tokenized_x = [sentence.lower().split() for sentence in self.raw_x]

        # Build vocab if not provided
        if vocab is None:
            all_words = [word for sentence in self.tokenized_x for word in sentence]
            freq = Counter(all_words)
            self.vocab = {word: idx+2 for idx, (word, _) in enumerate(freq.most_common())}
            self.vocab['<PAD>'] = 0
            self.vocab['<UNK>'] = 1
        else:
            self.vocab = vocab

    def encode(self, tokens):
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def __len__(self):
        return len(self.tokenized_x)

    def __getitem__(self, idx):
        tokens = self.tokenized_x[idx]
        ids = self.encode(tokens)[:self.max_len]
        padding = [self.vocab['<PAD>']] * (self.max_len - len(ids))
        input_ids = torch.tensor(ids + padding)
        label = torch.tensor(self.y[idx])
        return input_ids, label
