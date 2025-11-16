"""
Adversarial attack implementations for text classification.

This module implements two types of adversarial attacks:
1. KeywordSubstitutionAttack: Replaces important words with synonyms from embedding space
2. CharPerturbationAttack: Applies character-level edits to important words
"""

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
    Compute importance scores for each word using leave-one-out method.
    
    For each word, replace it with <unk> and measure the drop in true-label probability.
    Words that cause larger drops are more important for the prediction.
    
    Args:
        model: The text classification model
        input_ids: Token indices tensor of shape (seq_len,)
        true_label: Ground truth label
        pad_idx: Index of padding token
        unk_idx: Index of unknown token
        device: Device to run computations on
        
    Returns:
        Tuple of:
            - sorted_importance: List of (position, importance_score) sorted by score (descending)
            - base_prob: Original probability of true label before any perturbation
    """
    # Remember original training state and switch to eval mode
    was_training = model.training
    model.eval()
    
    try:
        # Get baseline prediction probability for the true label
        with torch.no_grad():
            x = input_ids.unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            base_prob = probs[true_label].item()

        # Find positions of non-padding tokens
        positions = [i for i, tok in enumerate(input_ids.tolist()) if tok != pad_idx]
        scores: List[Tuple[int, float]] = []

        # For each non-padding token, compute importance by leave-one-out
        for pos in positions:
            # Create perturbed input with token replaced by <unk>
            x_pert = input_ids.clone()
            x_pert[pos] = unk_idx
            
            # Compute probability drop when this token is removed
            with torch.no_grad():
                logits_pert = model(x_pert.unsqueeze(0).to(device))
                probs_pert = F.softmax(logits_pert, dim=1)[0]
                prob_true = probs_pert[true_label].item()
            
            # Importance is the drop in true-label probability
            drop = base_prob - prob_true
            scores.append((pos, drop))

        # Sort by importance score (descending)
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores, base_prob
        
    finally:
        # Restore original training state
        if was_training:
            model.train()


class KeywordSubstitutionAttack:
    """
    Greedy keyword substitution attack using embedding-space nearest neighbors.
    
    This attack:
    1. Identifies the most important words using leave-one-out scoring
    2. For each important word, finds semantically similar words (nearest neighbors in embedding space)
    3. Greedily substitutes words to maximize the drop in true-label probability
    4. Stops when misclassification occurs or budget is exhausted
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
        """
        Initialize the keyword substitution attack.
        
        Args:
            model: Text classification model
            vocab: Vocabulary object
            device: Device to run computations on
            pad_idx: Index of padding token
            unk_idx: Index of unknown token
            max_fraction_changed: Maximum fraction of tokens that can be changed (e.g., 0.2 = 20%)
            top_k_words: Number of most important words to consider for substitution
            max_synonyms: Maximum number of synonym candidates to consider per word
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.max_fraction_changed = max_fraction_changed
        self.top_k_words = top_k_words
        self.max_synonyms = max_synonyms

        random.seed(random_seed)
        self.refresh_embeddings()

    def refresh_embeddings(self):
        """
        Recompute normalized embedding matrix for cosine similarity.
        
        This should be called after model updates during adversarial training
        to ensure synonym candidates are based on current embeddings.
        """
        with torch.no_grad():
            # Get embedding matrix from model
            emb = self.model.emb.weight.detach()  # Shape: (vocab_size, emb_dim)
            
            # L2-normalize for cosine similarity computation
            norm = emb.norm(dim=1, keepdim=True) + 1e-8
            self.emb_norm = emb / norm

    def _is_attackable_token(self, token: str) -> bool:
        """
        Check if a token is suitable for attack.
        
        Filters out tokens that shouldn't be attacked:
        - Very short tokens (< 3 characters)
        - Pure punctuation
        - Tokens containing digits
        
        Args:
            token: Token string to check
            
        Returns:
            True if token can be attacked, False otherwise
        """
        if len(token) < 3:
            return False
        if all(ch in ".,!?;:-\"'`" for ch in token):
            return False
        if any(ch.isdigit() for ch in token):
            return False
        return True

    def _get_synonym_candidates(self, token_id: int) -> List[int]:
        """
        Get nearest-neighbor tokens in embedding space as synonym candidates.
        
        Args:
            token_id: Index of the token to find synonyms for
            
        Returns:
            List of token indices that are semantically similar
        """
        if token_id in (self.pad_idx, self.unk_idx):
            return []

        with torch.no_grad():
            # Get normalized embedding for this token
            vec = self.emb_norm[token_id]  # Shape: (emb_dim,)
            
            # Compute cosine similarity with all tokens
            sims = torch.mv(self.emb_norm, vec)  # Shape: (vocab_size,)

        # Get top-k most similar tokens
        k = min(self.max_synonyms + 5, sims.numel())
        topk = torch.topk(sims, k=k).indices.tolist()

        # Filter candidates
        candidates: List[int] = []
        for idx in topk:
            # Skip the original token
            if idx == token_id:
                continue
            # Skip special tokens
            if idx in (self.pad_idx, self.unk_idx):
                continue
            # Check if token is attackable
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
        Attack a single example by substituting important words.
        
        Args:
            input_ids: Token indices tensor of shape (seq_len,)
            tokens: List of token strings (same length as input_ids)
            true_label: Ground truth label
            max_changes: Maximum number of word substitutions allowed
            
        Returns:
            Tuple of:
                - adv_ids: Adversarial token indices
                - adv_tokens: Adversarial token strings
                - success: True if attack caused misclassification
                - num_changes: Number of words actually changed
        """
        assert input_ids.dim() == 1
        
        # Initialize adversarial example as copy of original
        orig_ids = input_ids.clone()
        adv_ids = input_ids.clone()
        adv_tokens = list(tokens)

        # Compute word importance scores
        importance, base_prob = compute_word_importance(
            self.model, adv_ids, true_label, self.pad_idx, self.unk_idx, self.device
        )

        # Focus on top-k most important words
        important_positions = [pos for pos, _ in importance][: self.top_k_words]

        # Compute maximum allowed changes based on fraction constraint
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

        # Greedily substitute important words
        for pos in important_positions:
            if num_changes >= max_allowed:
                break
                
            token_id = adv_ids[pos].item()
            token_str = adv_tokens[pos]

            # Skip non-attackable tokens
            if not self._is_attackable_token(token_str):
                continue

            # Get synonym candidates
            cand_ids = self._get_synonym_candidates(token_id)
            if not cand_ids:
                continue

            # Try each candidate and find the one that most reduces true-label probability
            best_cand = None
            best_prob = base_prob
            best_logits = None

            for cid in cand_ids:
                # Create trial input with this substitution
                trial_ids = adv_ids.clone()
                trial_ids[pos] = cid
                
                # Evaluate effect on model prediction
                with torch.no_grad():
                    logits = self.model(trial_ids.unsqueeze(0).to(self.device))
                    probs = F.softmax(logits, dim=1)[0]
                    prob_true = probs[true_label].item()
                
                # Keep track of best substitution
                if prob_true < best_prob:
                    best_prob = prob_true
                    best_cand = cid
                    best_logits = logits

            # If no improvement found, skip this word
            if best_cand is None:
                continue

            # Apply best substitution
            adv_ids[pos] = best_cand
            adv_tokens[pos] = self.vocab.itos[best_cand]
            num_changes += 1
            base_prob = best_prob

            # Check if attack succeeded (caused misclassification)
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
    Character-level perturbation attack.
    
    This attack applies small character-level edits (insert/delete/substitute/swap)
    to important words to fool the classifier while maintaining readability.
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
        """
        Initialize the character perturbation attack.
        
        Args:
            model: Text classification model
            vocab: Vocabulary object
            device: Device to run computations on
            pad_idx: Index of padding token
            unk_idx: Index of unknown token
            max_char_frac: Maximum fraction of characters that can be modified (e.g., 0.15 = 15%)
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.max_char_frac = max_char_frac
        random.seed(random_seed)

    def _is_attackable_token(self, token: str) -> bool:
        """
        Check if a token is suitable for character-level attack.
        
        Args:
            token: Token string to check
            
        Returns:
            True if token can be attacked, False otherwise
        """
        if len(token) < 3:
            return False
        if any(ch.isdigit() for ch in token):
            return False
        if all(ch in ".,!?;:-\"'`" for ch in token):
            return False
        return True

    def _apply_edit(self, word: str) -> str:
        """
        Apply a random character-level edit to a word.
        
        Possible edits:
        - Substitute: Replace a character with a random letter
        - Delete: Remove a character
        - Insert: Add a random character at a position
        - Swap: Transpose two adjacent characters
        
        Args:
            word: Input word string
            
        Returns:
            Modified word string
        """
        if len(word) == 0:
            return word
            
        # Randomly choose edit operation
        op = random.choice(["substitute", "delete", "insert", "swap"])
        chars = list(word)
        
        if op == "substitute":
            # Replace a random character
            idx = random.randrange(len(chars))
            new_ch = random.choice(self.ALPHABET)
            chars[idx] = new_ch
            
        elif op == "delete" and len(chars) > 1:
            # Remove a random character
            idx = random.randrange(len(chars))
            del chars[idx]
            
        elif op == "insert":
            # Insert a random character at a random position
            idx = random.randrange(len(chars) + 1)
            new_ch = random.choice(self.ALPHABET)
            chars.insert(idx, new_ch)
            
        elif op == "swap" and len(chars) > 1:
            # Swap two adjacent characters
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
        """
        Attack a single example by applying character-level edits.
        
        Args:
            input_ids: Token indices tensor of shape (seq_len,)
            tokens: List of token strings (same length as input_ids)
            true_label: Ground truth label
            max_edits: Maximum number of character edits allowed
            
        Returns:
            Tuple of:
                - adv_ids: Adversarial token indices
                - adv_tokens: Adversarial token strings
                - success: True if attack caused misclassification
                - num_edits: Number of edits actually applied
        """
        # Compute word importance scores
        importance, base_prob = compute_word_importance(
            self.model, input_ids, true_label, self.pad_idx, self.unk_idx, self.device
        )
        important_positions = [pos for pos, _ in importance]

        # Compute maximum allowed edits based on character fraction constraint
        real_positions = [i for i, tok in enumerate(input_ids.tolist()) if tok != self.pad_idx]
        total_chars = sum(len(tokens[i]) for i in real_positions)
        max_allowed_by_frac = int(math.ceil(self.max_char_frac * max(total_chars, 1)))
        max_allowed = max(1, min(max_edits, max_allowed_by_frac))

        # Initialize adversarial example
        adv_tokens = list(tokens)
        num_edits = 0
        success = False
        adv_ids = input_ids.clone()

        # Greedily apply character edits to important words
        for pos in important_positions:
            if num_edits >= max_allowed:
                break
            if pos >= len(adv_tokens):
                continue
                
            word = adv_tokens[pos]
            
            # Skip non-attackable tokens
            if not self._is_attackable_token(word):
                continue

            # Apply a random character edit
            new_word = self._apply_edit(word)
            tmp_tokens = list(adv_tokens)
            tmp_tokens[pos] = new_word

            # Encode modified tokens and evaluate effect
            encoded = self.vocab.encode(tmp_tokens, max_len=len(input_ids))
            trial_ids = torch.tensor(encoded, dtype=torch.long)

            with torch.no_grad():
                logits = self.model(trial_ids.unsqueeze(0).to(self.device))
                probs = F.softmax(logits, dim=1)[0]
                prob_true = probs[true_label].item()

            # Keep edits that hurt the model's confidence
            if prob_true < base_prob:
                adv_tokens[pos] = new_word
                adv_ids = trial_ids
                base_prob = prob_true
                num_edits += 1

                # Check if attack succeeded
                pred_label = probs.argmax().item()
                if pred_label != true_label:
                    success = True
                    break

        return adv_ids, adv_tokens, success, num_edits
