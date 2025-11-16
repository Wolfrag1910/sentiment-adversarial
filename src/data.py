"""
Data loading and preprocessing for IMDB sentiment classification.

This module handles reading the IMDB dataset, text cleaning, tokenization,
vocabulary building, and creating PyTorch DataLoaders.
"""

import os
import glob
import html
import random
import unicodedata
import re
from torch.utils.data import Dataset, DataLoader
from .vocab import tokenize, Vocab
import torch

# Regular expression to match HTML break tags
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def clean_text(s, lower=True, rm_br=True, norm=True):
    """
    Clean and normalize text data.
    
    Args:
        s: Input text string
        lower: Convert to lowercase if True
        rm_br: Remove HTML break tags if True
        norm: Apply Unicode normalization (NFKC) if True
        
    Returns:
        Cleaned text string
    """
    # Normalize Unicode characters to standard form
    if norm:
        s = unicodedata.normalize("NFKC", s)
    
    # Decode HTML entities (e.g., &amp; -> &)
    s = html.unescape(s)
    
    # Remove HTML break tags
    if rm_br:
        s = BR_RE.sub(" ", s)
    
    # Convert to lowercase
    return s.lower() if lower else s


def read_split(root, split):
    """
    Read a split (train/test) of the IMDB dataset.
    
    Args:
        root: Root directory of IMDB dataset (e.g., 'data/raw/aclImdb')
        split: Split name ('train' or 'test')
        
    Returns:
        List of (text, label) tuples where label is 1 for positive, 0 for negative
    """
    # Read positive reviews
    pos = glob.glob(os.path.join(root, split, "pos", "*.txt"))
    # Read negative reviews
    neg = glob.glob(os.path.join(root, split, "neg", "*.txt"))
    
    # Create dataset with labels (1 for positive, 0 for negative)
    data = [(open(p, encoding="utf-8").read(), 1) for p in pos] + \
           [(open(n, encoding="utf-8").read(), 0) for n in neg]
    
    # Shuffle to mix positive and negative examples
    random.shuffle(data)
    return data


class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    
    Stores pre-encoded text sequences and their labels.
    """
    
    def __init__(self, texts, labels):
        """
        Initialize dataset with encoded texts and labels.
        
        Args:
            texts: List of encoded sequences (lists of token indices)
            labels: List of integer labels
        """
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tuple of (text_tensor, label_tensor)
        """
        x = torch.tensor(self.texts[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def build_imdb_splits(cfg):
    """
    Build train/validation/test splits with vocabulary for IMDB dataset.
    
    Args:
        cfg: Configuration dictionary with data processing parameters
        
    Returns:
        Tuple of:
            - (train_X, train_y): Training data and labels
            - (val_X, val_y): Validation data and labels
            - (test_X, test_y): Test data and labels (or None if not using official test)
            - V: Vocabulary object
            - meta: Dictionary with metadata (vocab_size, etc.)
    """
    # Read raw data from disk
    train_all = read_split(cfg["data"]["root"], "train")
    test = read_split(cfg["data"]["root"], "test") if cfg["data"]["use_official_test"] else None
    
    # Split training data into train and validation sets
    val_ratio = cfg["data"]["val_ratio"]
    n_train = int((1 - val_ratio) * len(train_all))
    train = train_all[:n_train]
    val = train_all[n_train:]

    def prep(batch):
        """
        Clean, tokenize, and extract labels from a batch of examples.
        
        Args:
            batch: List of (text, label) tuples
            
        Returns:
            Tuple of (tokenized_texts, labels)
        """
        # Clean text with configured preprocessing options
        texts = [clean_text(x, cfg["data"]["lowercase"], 
                           cfg["data"]["remove_html_breaks"],
                           cfg["data"]["normalize_unicode"]) for x, _ in batch]
        
        # Tokenize cleaned text
        toks = [tokenize(x) for x in texts]
        
        # Extract labels
        labs = [y for _, y in batch]
        return toks, labs

    # Preprocess all splits
    tr_tok, tr_y = prep(train)
    va_tok, va_y = prep(val)
    te_tok, te_y = prep(test) if test else (None, None)

    # Build vocabulary from training data only (to prevent test set leakage)
    V = Vocab(cfg["data"]["min_freq"], cfg["data"]["max_vocab"],
              cfg["data"]["pad_token"], cfg["data"]["unk_token"])
    V.build(tr_tok)

    def encode(toks):
        """
        Encode tokenized texts to fixed-length sequences of indices.
        
        Args:
            toks: List of token lists
            
        Returns:
            List of encoded sequences (lists of indices)
        """
        return [V.encode(t, cfg["data"]["max_len"]) for t in toks]
    
    # Encode all splits using the vocabulary
    tr_X = encode(tr_tok)
    va_X = encode(va_tok)
    te_X = encode(te_tok) if te_tok else None

    # Create metadata dictionary
    meta = {"vocab_size": len(V.itos)}
    
    return (tr_X, tr_y), (va_X, va_y), (te_X, te_y), V, meta


def make_loaders(tr, va, cfg):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        tr: Tuple of (train_X, train_y)
        va: Tuple of (val_X, val_y)
        cfg: Configuration dictionary with batch size and num_workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create Dataset objects
    tr_ds = TextDataset(tr[0], tr[1])
    va_ds = TextDataset(va[0], va[1])
    
    # Get batch size from config
    bs = cfg["data"]["batch_size"]
    
    # Create DataLoaders with shuffling for training, no shuffling for validation
    train_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, 
                             num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, 
                           num_workers=cfg["data"]["num_workers"])
    
    return train_loader, val_loader
