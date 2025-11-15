import math
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F


def compute_word_importance(
    model,
    input_ids: torch.Tensor,
    true_label: int,
    pad_idx: int,
    unk_idx: int,
    device: torch.device,
) -> Tuple[List[Tuple[int, float]], float]:
    """
    Leave-one-out importance: replace each token with <unk> and measure drop in
    true-label probability.

    input_ids: (T,) tensor of token indices (already padded to max_len)
    Returns:
        sorted_importance: list of (position, score) sorted desc by score
        base_prob: p(y=true_label | x)
    """
    model.eval()
    with torch.no_grad():
        x = input_ids.unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        base_prob = probs[true_label].item()

    # positions of non-pad tokens
    positions = [i for i, tok in enumerate(input_ids.tolist()) if tok != pad_idx]

    scores: List[Tuple[int, float]] = []
    for pos in positions:
        x_pert = input_ids.clone()
        x_pert[pos] = unk_idx
        with torch.no_grad():
            logits_pert = model(x_pert.unsqueeze(0).to(device))
            probs_pert = F.softmax(logits_pert, dim=1)[0]
            prob_true = probs_pert[true_label].item()
        drop = base_prob - prob_true
        scores.append((pos, drop))

    scores.sort(key=lambda t: t[1], reverse=True)
    return scores, base_prob


class KeywordSubstitutionAttack:
    """
    Greedy keyword-substitution attack using nearest neighbours in the model's
    embedding space as candidate replacements.
    """

    def __init__(
        self,
        model,
        vocab,
        device,
        pad_idx: int = 0,
        unk_idx: int = 1,
        max_fraction_changed: float = 0.2,
        top_k_words: int = 8,
        max_synonyms: int = 20,
        random_seed: int = 1337,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.max_fraction_changed = max_fraction_changed
        self.top_k_words = top_k_words
        self.max_synonyms = max_synonyms

        random.seed(random_seed)

        # Precompute L2-normalised embedding matrix for cosine similarity
        with torch.no_grad():
            emb = model.emb.weight.detach()  # (V, D)
            norm = emb.norm(dim=1, keepdim=True) + 1e-8
            self.emb_norm = emb / norm

    def _is_attackable_token(self, token: str) -> bool:
        # skip padding, unk, very short tokens, purely punctuation or digits
        if len(token) < 3:
            return False
        if all(ch in ".,!?;:-\"'`" for ch in token):
            return False
        if any(ch.isdigit() for ch in token):
            return False
        return True

    def _get_synonym_candidates(self, token_id: int) -> List[int]:
        """
        Get nearest-neighbour tokens for the given token in embedding space.
        """
        if token_id in (self.pad_idx, self.unk_idx):
            return []

        with torch.no_grad():
            vec = self.emb_norm[token_id]  # (D,)
            sims = torch.mv(self.emb_norm, vec)  # (V,)

        # Get top max_synonyms+5 to allow filtering
        k = min(self.max_synonyms + 5, sims.numel())
        topk = torch.topk(sims, k=k).indices.tolist()

        candidates: List[int] = []
        for idx in topk:
            if idx == token_id:
                continue
            if idx in (self.pad_idx, self.unk_idx):
                continue
            tok = self.vocab.itos[idx]
            if not self._is_attackable_token(tok):
                continue
            candidates.append(idx)
            if len(candidates) >= self.max_synonyms:
                break
        return candidates

    def attack_example(
        self,
        input_ids: torch.Tensor,
        tokens: List[str],
        true_label: int,
        max_changes: int,
    ) -> Tuple[torch.Tensor, List[str], bool, int]:
        """
        Attack a single example.

        input_ids: (T,) tensor
        tokens: list[str] of same length (unpadded tokens followed by pads)
        Returns:
            adv_ids, adv_tokens, success, num_changes
        """
        assert input_ids.dim() == 1
        orig_ids = input_ids.clone()
        adv_ids = input_ids.clone()
        adv_tokens = list(tokens)

        importance, base_prob = compute_word_importance(
            self.model, adv_ids, true_label, self.pad_idx, self.unk_idx, self.device
        )

        # limit to top_k_words
        important_positions = [pos for pos, _ in importance][: self.top_k_words]

        # change budget by fraction of real tokens
        real_positions = [i for i, tok in enumerate(orig_ids.tolist()) if tok != self.pad_idx]
        max_allowed = max(
            1,
            min(
                max_changes,
                int(math.ceil(self.max_fraction_changed * len(real_positions))),
            ),
        )

        num_changes = 0
        success = False

        for pos in important_positions:
            if num_changes >= max_allowed:
                break
            token_id = adv_ids[pos].item()
            token_str = adv_tokens[pos]

            if not self._is_attackable_token(token_str):
                continue

            cand_ids = self._get_synonym_candidates(token_id)
            if not cand_ids:
                continue

            best_cand = None
            best_prob = base_prob
            best_logits = None

            for cid in cand_ids:
                trial_ids = adv_ids.clone()
                trial_ids[pos] = cid
                with torch.no_grad():
                    logits = self.model(trial_ids.unsqueeze(0).to(self.device))
                    probs = F.softmax(logits, dim=1)[0]
                    prob_true = probs[true_label].item()
                if prob_true < best_prob:
                    best_prob = prob_true
                    best_cand = cid
                    best_logits = logits

            if best_cand is None:
                continue

            # apply best substitution
            adv_ids[pos] = best_cand
            adv_tokens[pos] = self.vocab.itos[best_cand]
            num_changes += 1
            base_prob = best_prob

            # check for success
            if best_logits is None:
                with torch.no_grad():
                    best_logits = self.model(adv_ids.unsqueeze(0).to(self.device))
            pred_label = best_logits.argmax(dim=1).item()
            if pred_label != true_label:
                success = True
                break

        return adv_ids, adv_tokens, success, num_changes


class CharPerturbationAttack:
    """
    Simple character-level attack that applies a small number of edits
    (insert/delete/substitute/swap) to important words.
    """

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def __init__(
        self,
        model,
        vocab,
        device,
        pad_idx: int = 0,
        unk_idx: int = 1,
        max_char_frac: float = 0.15,
        random_seed: int = 1337,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.max_char_frac = max_char_frac
        random.seed(random_seed)

    def _is_attackable_token(self, token: str) -> bool:
        if len(token) < 3:
            return False
        if any(ch.isdigit() for ch in token):
            return False
        if all(ch in ".,!?;:-\"'`" for ch in token):
            return False
        return True

    def _apply_edit(self, word: str) -> str:
        if len(word) == 0:
            return word
        op = random.choice(["substitute", "delete", "insert", "swap"])
        chars = list(word)
        if op == "substitute":
            idx = random.randrange(len(chars))
            new_ch = random.choice(self.ALPHABET)
            chars[idx] = new_ch
        elif op == "delete" and len(chars) > 1:
            idx = random.randrange(len(chars))
            del chars[idx]
        elif op == "insert":
            idx = random.randrange(len(chars) + 1)
            new_ch = random.choice(self.ALPHABET)
            chars.insert(idx, new_ch)
        elif op == "swap" and len(chars) > 1:
            idx = random.randrange(len(chars) - 1)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def attack_example(
        self,
        input_ids: torch.Tensor,
        tokens: List[str],
        true_label: int,
        max_edits: int,
    ) -> Tuple[torch.Tensor, List[str], bool, int]:
        # initial importance
        importance, base_prob = compute_word_importance(
            self.model, input_ids, true_label, self.pad_idx, self.unk_idx, self.device
        )
        important_positions = [pos for pos, _ in importance]

        # compute total characters in real tokens
        real_positions = [i for i, tok in enumerate(input_ids.tolist()) if tok != self.pad_idx]
        total_chars = sum(len(tokens[i]) for i in real_positions)
        max_allowed_by_frac = int(math.ceil(self.max_char_frac * max(total_chars, 1)))
        max_allowed = max(1, min(max_edits, max_allowed_by_frac))

        adv_tokens = list(tokens)
        num_edits = 0
        success = False
        adv_ids = input_ids.clone()

        for pos in important_positions:
            if num_edits >= max_allowed:
                break
            if pos >= len(adv_tokens):
                continue
            word = adv_tokens[pos]
            if not self._is_attackable_token(word):
                continue

            new_word = self._apply_edit(word)
            tmp_tokens = list(adv_tokens)
            tmp_tokens[pos] = new_word

            # encode with vocab
            encoded = self.vocab.encode(tmp_tokens, max_len=len(input_ids))
            trial_ids = torch.tensor(encoded, dtype=torch.long)

            with torch.no_grad():
                logits = self.model(trial_ids.unsqueeze(0).to(self.device))
                probs = F.softmax(logits, dim=1)[0]
                prob_true = probs[true_label].item()

            if prob_true < base_prob:  # keep edits that hurt the model
                adv_tokens[pos] = new_word
                adv_ids = trial_ids
                base_prob = prob_true
                num_edits += 1

                pred_label = probs.argmax().item()
                if pred_label != true_label:
                    success = True
                    break

        return adv_ids, adv_tokens, success, num_edits
