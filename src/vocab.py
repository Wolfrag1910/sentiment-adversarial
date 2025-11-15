from collections import Counter
import re

WS_RE = re.compile(r"\w+|[^\w\s]")  # tokens + punctuation

def tokenize(text, lowercase=True):
    if lowercase: text = text.lower()
    return WS_RE.findall(text)

class Vocab:
    def __init__(self, min_freq=1, max_size=None, pad="<pad>", unk="<unk>"):
        self.min_freq, self.max_size = min_freq, max_size
        self.pad, self.unk = pad, unk
        self.itos, self.stoi = [], {}

    def build(self, texts):
        freq = Counter()
        for t in texts: freq.update(t)
        words = [w for w,c in freq.items() if c >= self.min_freq]
        words.sort(key=lambda w: (-freq[w], w))
        if self.max_size: words = words[:self.max_size]
        self.itos = [self.pad, self.unk] + words
        self.stoi = {w:i for i,w in enumerate(self.itos)}

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, self.stoi[self.unk]) for t in tokens][:max_len]
        if len(ids) < max_len:
            ids += [self.stoi[self.pad]] * (max_len - len(ids))
        return ids