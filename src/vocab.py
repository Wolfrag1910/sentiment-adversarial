"""
Vocabulary management and tokenization utilities.

This module handles text tokenization and vocabulary building for the text classification model.
"""

from collections import Counter
import re

# Regular expression to extract words and punctuation as separate tokens
WS_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text, lowercase=True):
    """
    Tokenize text into words and punctuation marks.
    
    Args:
        text: Input text string
        lowercase: Whether to convert text to lowercase before tokenization
        
    Returns:
        List of tokens (words and punctuation)
    """
    if lowercase:
        text = text.lower()
    return WS_RE.findall(text)


class Vocab:
    """
    Vocabulary class for mapping tokens to indices and vice versa.
    
    Builds a vocabulary from training data with support for minimum frequency
    filtering, maximum vocabulary size, and special tokens (padding, unknown).
    """
    
    def __init__(self, min_freq=1, max_size=None, pad="<pad>", unk="<unk>"):
        """
        Initialize vocabulary with configuration parameters.
        
        Args:
            min_freq: Minimum frequency for a token to be included in vocabulary
            max_size: Maximum vocabulary size (None for unlimited)
            pad: Padding token string
            unk: Unknown token string for out-of-vocabulary words
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.pad = pad
        self.unk = unk
        self.itos = []  # Index to string mapping
        self.stoi = {}  # String to index mapping
        self.freqs = {}  # Token frequency counts

    def build(self, texts):
        """
        Build vocabulary from a list of tokenized texts.
        
        Args:
            texts: List of token lists (e.g., [['hello', 'world'], ['foo', 'bar']])
        """
        # Count token frequencies across all texts
        freq = Counter()
        for t in texts:
            freq.update(t)
        
        # Filter tokens by minimum frequency
        words = [w for w, c in freq.items() if c >= self.min_freq]
        
        # Sort by frequency (descending) then alphabetically for determinism
        words.sort(key=lambda w: (-freq[w], w))
        
        # Limit vocabulary size if specified
        if self.max_size:
            words = words[:self.max_size]
        
        # Build index mappings with special tokens at the beginning
        self.itos = [self.pad, self.unk] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        # Store token frequencies for spell correction
        # Give padding token a very high count so it's never "corrected"
        self.freqs = {w: freq.get(w, 0) for w in self.itos}
        self.freqs[self.pad] = 10**9
        self.freqs[self.unk] = 1

    def encode(self, tokens, max_len):
        """
        Convert tokens to indices with padding/truncation to fixed length.
        
        Args:
            tokens: List of token strings
            max_len: Maximum sequence length (truncate if longer, pad if shorter)
            
        Returns:
            List of token indices of length max_len
        """
        # Convert tokens to indices, using <unk> for out-of-vocabulary tokens
        ids = [self.stoi.get(t, self.stoi[self.unk]) for t in tokens][:max_len]
        
        # Pad sequence to max_len if necessary
        if len(ids) < max_len:
            ids += [self.stoi[self.pad]] * (max_len - len(ids))
        
        return ids

    def in_vocab(self, token: str) -> bool:
        """
        Check if a token exists in the vocabulary.
        
        Args:
            token: Token string to check
            
        Returns:
            True if token is in vocabulary, False otherwise
        """
        return token in self.stoi

    def freq(self, token: str) -> int:
        """
        Get the training frequency of a token.
        
        Args:
            token: Token string
            
        Returns:
            Frequency count (0 if token not in vocabulary)
        """
        return self.freqs.get(token, 0)
