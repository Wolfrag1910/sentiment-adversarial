from collections import Counter
import re

WS_RE = re.compile(r"\w+|[^\w\s]")

def tokenize(text, lowercase=True):
    if lowercase: text = text.lower()
    return WS_RE.findall(text)

class Vocab:
    def __init__(self, min_freq=1, max_size=None, pad="<pad>", unk="<unk>"):
        self.min_freq, self.max_size = min_freq, max_size
        self.pad, self.unk = pad, unk
        self.itos, self.stoi = [], {}
        self.freqs = {}

    def build(self, texts):
        freq = Counter()
        for t in texts: freq.update(t)
        words = [w for w, c in freq.items() if c >= self.min_freq]
        words.sort(key=lambda w: (-freq[w], w))
        if self.max_size: words = words[:self.max_size]
        self.itos = [self.pad, self.unk] + words
        self.stoi = {w:i for i,w in enumerate(self.itos)}

        # store frequencies; give pad a huge count so it's never “corrected”
        self.freqs = {w: freq.get(w, 0) for w in self.itos}
        self.freqs[self.pad] = 10**9
        self.freqs[self.unk] = 1

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, self.stoi[self.unk]) for t in tokens][:max_len]
        if len(ids) < max_len:
            ids += [self.stoi[self.pad]] * (max_len - len(ids))
        return ids

    def in_vocab(self, token: str) -> bool:
        return token in self.stoi

    def freq(self, token: str) -> int:
        return self.freqs.get(token, 0)
