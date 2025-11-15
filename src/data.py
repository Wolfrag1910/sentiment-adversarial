import os, glob, html, random, unicodedata, re, json
from torch.utils.data import Dataset, DataLoader
from .vocab import tokenize, Vocab
import torch

BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)

def clean_text(s, lower=True, rm_br=True, norm=True):
    if norm: s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    if rm_br: s = BR_RE.sub(" ", s)
    return s.lower() if lower else s

def read_split(root, split):
    pos = glob.glob(os.path.join(root, split, "pos", "*.txt"))
    neg = glob.glob(os.path.join(root, split, "neg", "*.txt"))
    data = [(open(p, encoding="utf-8").read(), 1) for p in pos] + \
           [(open(n, encoding="utf-8").read(), 0) for n in neg]
    random.shuffle(data)
    return data

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        x = torch.tensor(self.texts[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def build_imdb_splits(cfg):
    train_all = read_split(cfg["data"]["root"], "train")
    test = read_split(cfg["data"]["root"], "test") if cfg["data"]["use_official_test"] else None
    # train/val split
    val_ratio = cfg["data"]["val_ratio"]
    n_train = int((1 - val_ratio) * len(train_all))
    train = train_all[:n_train]; val = train_all[n_train:]

    # clean + tokenize
    def prep(batch):
        texts = [clean_text(x, cfg["data"]["lowercase"], cfg["data"]["remove_html_breaks"],
                            cfg["data"]["normalize_unicode"]) for x,_ in batch]
        toks  = [tokenize(x) for x in texts]
        labs  = [y for _,y in batch]
        return toks, labs

    tr_tok, tr_y = prep(train)
    va_tok, va_y = prep(val)
    te_tok, te_y = prep(test) if test else (None, None)

    # vocab on train only
    V = Vocab(cfg["data"]["min_freq"], cfg["data"]["max_vocab"],
              cfg["data"]["pad_token"], cfg["data"]["unk_token"])
    V.build(tr_tok)

    # encode & pad
    def encode(toks): return [V.encode(t, cfg["data"]["max_len"]) for t in toks]
    tr_X = encode(tr_tok); va_X = encode(va_tok)
    te_X = encode(te_tok) if te_tok else None

    meta = {"vocab_size": len(V.itos)}
    return (tr_X, tr_y), (va_X, va_y), (te_X, te_y), V, meta

def make_loaders(tr, va, cfg):
    import torch
    tr_ds = TextDataset(tr[0], tr[1])
    va_ds = TextDataset(va[0], va[1])
    bs = cfg["data"]["batch_size"]
    return (DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=cfg["data"]["num_workers"]),
            DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=cfg["data"]["num_workers"]))