import json
import torch
from torch.utils.data import Dataset
from collections import Counter

class Sent140Dataset(Dataset):
    def __init__(self, json_path, user, vocab=None, max_len=50):
        with open(json_path, 'r') as f:
            data = json.load(f)['user_data'][user]

        self.raw_x = data['x']
        self.y = data['y']
        self.max_len = max_len

        # Tokenization
        self.tokenized_x = [sentence.lower().split() for sentence in self.raw_x]

        # Build or assign vocabulary
        if vocab is None:
            all_words = [word for sent in self.tokenized_x for word in sent]
            freq = Counter(all_words)
            self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(freq.most_common())}
            self.vocab['<PAD>'] = 0
            self.vocab['<UNK>'] = 1
        else:
            self.vocab = vocab

    def encode(self, tokens):
        return [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]

    def __getitem__(self, idx):
        tokens = self.tokenized_x[idx]
        ids = self.encode(tokens)[:self.max_len]
        padded = ids + [self.vocab['<PAD>']] * (self.max_len - len(ids))
        return torch.tensor(padded), torch.tensor(self.y[idx])

    def __len__(self):
        return len(self.tokenized_x)
