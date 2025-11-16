import unicodedata
import re
import string
from typing import List

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

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
HANDLE_RE = re.compile(r"^@[\w_]+$")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # 3+ repeated chars


def normalize_token(tok: str) -> str:
    # NFKC + lowercase
    tok = unicodedata.normalize("NFKC", tok)
    tok = tok.lower()
    # collapse “cooool” -> “cool”
    tok = REPEAT_RE.sub(r"\1\1", tok)
    # basic leetspeak normalisation
    tok = tok.translate(L33T_MAP)
    return tok


def edits1(word: str) -> set[str]:
    """All edit-distance-1 candidates (Norvig-style)."""
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + (R[1:] if len(R) else "") for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def best_correction(token: str, vocab, max_candidates: int = 100) -> str:
    """
    Very simple spell-correction:
      - generate edit-distance-1 candidates
      - keep those in vocab
      - pick the one with highest training frequency
    """
    if vocab.in_vocab(token):
        return token

    if len(token) > 15 or len(token) < 3:
        return token

    candidates = []
    for cand in edits1(token):
        if vocab.in_vocab(cand):
            candidates.append(cand)
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        return token

    # pick candidate with highest training frequency
    best = max(candidates, key=lambda w: vocab.freq(w))
    return best


def sanitize_tokens(tokens: List[str], vocab) -> List[str]:
    """
    Input-sanitization pipeline over tokens.

    Steps:
      - normalize Unicode + lowercase
      - strip obvious URLs and @handles to special tokens
      - collapse repeated characters
      - normalise leetspeak
      - correct OOV tokens by edit-distance-1 to closest in-vocab word
    """
    out: List[str] = []
    for tok in tokens:
        # basic URL / handle stripping
        if URL_RE.match(tok):
            out.append("<url>")
            continue
        if HANDLE_RE.match(tok):
            out.append("<user>")
            continue

        ntok = normalize_token(tok)

        # keep punctuation as-is (after normalization)
        if all(ch in ".,!?;:-\"'`()[]{}" for ch in ntok):
            out.append(ntok)
            continue

        # if not in vocab, try correction
        if not vocab.in_vocab(ntok):
            ntok = best_correction(ntok, vocab)

        out.append(ntok)

    return out
