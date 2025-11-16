"""
Input sanitization defense for adversarial text attacks.

This module implements preprocessing-based defenses that normalize and correct
adversarial perturbations before they reach the model.
"""

import unicodedata
import re
import string
from typing import List

# Leetspeak character mapping for normalization
L33T_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
})

# Regular expressions for detecting URLs and social media handles
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
HANDLE_RE = re.compile(r"^@[\w_]+$")

# Pattern for detecting repeated characters (3 or more)
REPEAT_RE = re.compile(r"(.)\1{2,}")


def normalize_token(tok: str) -> str:
    """
    Normalize a single token to reduce adversarial perturbations.
    
    Applies multiple normalization steps:
    1. Unicode normalization (NFKC)
    2. Lowercasing
    3. Collapse repeated characters (e.g., "cooool" -> "cool")
    4. Leetspeak normalization (e.g., "h3ll0" -> "hello")
    
    Args:
        tok: Input token string
        
    Returns:
        Normalized token string
    """
    # Apply Unicode normalization to handle various character encodings
    tok = unicodedata.normalize("NFKC", tok)
    
    # Convert to lowercase
    tok = tok.lower()
    
    # Collapse repeated characters (keep at most 2 repetitions)
    # This handles attacks like "amaziiiiing" -> "amazing"
    tok = REPEAT_RE.sub(r"\1\1", tok)
    
    # Normalize basic leetspeak substitutions
    tok = tok.translate(L33T_MAP)
    
    return tok


def edits1(word: str) -> set[str]:
    """
    Generate all strings that are one edit away from the input word.
    
    Uses the Norvig spelling correction algorithm to generate candidates:
    - Deletions: Remove one character
    - Transpositions: Swap two adjacent characters
    - Replacements: Change one character to any letter
    - Insertions: Add one character at any position
    
    Args:
        word: Input word string
        
    Returns:
        Set of all possible one-edit variations
    """
    letters = string.ascii_lowercase
    
    # Split word at each position
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    # Generate all possible edits
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + (R[1:] if len(R) else "") for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    
    return set(deletes + transposes + replaces + inserts)


def best_correction(token: str, vocab, max_candidates: int = 100) -> str:
    """
    Correct a potentially misspelled token using edit-distance-1 candidates.
    
    This helps defend against character-level perturbation attacks by
    correcting typos back to in-vocabulary words.
    
    Algorithm:
    1. If token is already in vocabulary, return it unchanged
    2. Generate all edit-distance-1 candidates
    3. Filter to only in-vocabulary candidates
    4. Return the candidate with highest training frequency
    
    Args:
        token: Token to potentially correct
        vocab: Vocabulary object with in_vocab() and freq() methods
        max_candidates: Maximum number of candidates to consider
        
    Returns:
        Corrected token (or original if no good correction found)
    """
    # If token is already in vocabulary, no correction needed
    if vocab.in_vocab(token):
        return token

    # Skip very long or very short tokens (unlikely to be correctable)
    if len(token) > 15 or len(token) < 3:
        return token

    # Generate edit-distance-1 candidates
    candidates = []
    for cand in edits1(token):
        if vocab.in_vocab(cand):
            candidates.append(cand)
        if len(candidates) >= max_candidates:
            break

    # If no valid candidates found, return original token
    if not candidates:
        return token

    # Return the candidate with highest training frequency
    # This assumes more frequent words are more likely to be correct
    best = max(candidates, key=lambda w: vocab.freq(w))
    return best


def sanitize_tokens(tokens: List[str], vocab) -> List[str]:
    """
    Apply input sanitization pipeline to a list of tokens.
    
    This is the main defense function that applies multiple preprocessing steps
    to reduce the effectiveness of adversarial perturbations:
    
    1. Strip URLs and social media handles to special tokens
    2. Normalize Unicode, lowercase, collapse repeated chars, fix leetspeak
    3. Correct out-of-vocabulary tokens using edit-distance-1 spell correction
    
    Args:
        tokens: List of input token strings
        vocab: Vocabulary object for spell correction
        
    Returns:
        List of sanitized token strings
    """
    out: List[str] = []
    
    for tok in tokens:
        # Replace URLs with special token
        if URL_RE.match(tok):
            out.append("<url>")
            continue
            
        # Replace social media handles with special token
        if HANDLE_RE.match(tok):
            out.append("<user>")
            continue

        # Apply normalization (Unicode, lowercase, repeated chars, leetspeak)
        ntok = normalize_token(tok)

        # Keep punctuation as-is after normalization
        if all(ch in ".,!?;:-\"'`()[]{}" for ch in ntok):
            out.append(ntok)
            continue

        # If token is not in vocabulary, try to correct it
        # This defends against character-level perturbation attacks
        if not vocab.in_vocab(ntok):
            ntok = best_correction(ntok, vocab)

        out.append(ntok)

    return out
