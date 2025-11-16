"""
Adversarial attack evaluation script.

Evaluates the robustness of trained models against keyword substitution and
character perturbation attacks, with optional input sanitization defense.
"""

import argparse
import csv
import os
import pickle
from typing import Dict, Any, List

import torch
import yaml

from src.data import read_split, clean_text
from src.vocab import tokenize
from src.models.cnn_text import TextCNN
from src.attacks.text_attacks import KeywordSubstitutionAttack, CharPerturbationAttack
from src.defenses.sanitization import sanitize_tokens


def load_cfg(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_vocab(cfg, ckpt_path):
    """
    Load trained model and vocabulary.
    
    Args:
        cfg: Configuration dictionary
        ckpt_path: Path to model checkpoint
        
    Returns:
        Tuple of (model, vocab, device)
    """
    # Set up device (CPU or GPU)
    if cfg["project"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["project"]["device"])

    # Load vocabulary saved during training
    vocab_path = os.path.join(cfg["log"]["dir"], "vocab.pkl")
    with open(vocab_path, "rb") as f:
        V = pickle.load(f)

    # Build model
    model = TextCNN(
        vocab_size=len(V.itos),
        emb_dim=cfg["model"]["embedding_dim"],
        num_classes=cfg["model"]["num_classes"],
        filter_sizes=cfg["model"]["filter_sizes"],
        num_filters=cfg["model"]["num_filters"],
        dropout=cfg["model"]["dropout"],
        pad_idx=0,
    ).to(device)
    
    # Load trained weights
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    return model, V, device


def prepare_dataset(cfg):
    """
    Load and preprocess test dataset.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (tokens_list, labels)
    """
    root = cfg["data"]["root"]
    test_raw = read_split(root, "test")

    # Clean text with same settings as training
    lower = cfg["data"]["lowercase"]
    rm_br = cfg["data"]["remove_html_breaks"]
    norm = cfg["data"]["normalize_unicode"]

    texts = [clean_text(x, lower, rm_br, norm) for x, _ in test_raw]
    labels = [y for _, y in test_raw]
    tokens_list = [tokenize(t) for t in texts]
    
    return tokens_list, labels


def encode_tokens(tokens: List[str], vocab, max_len: int) -> List[int]:
    """
    Encode tokens to indices with padding/truncation.
    
    Args:
        tokens: List of token strings
        vocab: Vocabulary object
        max_len: Maximum sequence length
        
    Returns:
        List of token indices
    """
    return vocab.encode(tokens, max_len=max_len)


def evaluate_attack_on_dataset(
    attack_name: str,
    attack_obj,
    tokens_list,
    labels,
    vocab,
    device,
    max_change: int,
    max_examples: int | None,
    max_len: int,
    use_sanitization: bool,
):
    """
    Evaluate an attack on the test dataset.
    
    Args:
        attack_name: Name of the attack for logging
        attack_obj: Attack object (KeywordSubstitutionAttack or CharPerturbationAttack)
        tokens_list: List of tokenized test examples
        labels: List of test labels
        vocab: Vocabulary object
        device: Device to run computations on
        max_change: Attack budget (max number of changes)
        max_examples: Maximum number of examples to evaluate (None for all)
        max_len: Maximum sequence length
        use_sanitization: Whether to apply input sanitization defense
        
    Returns:
        Dictionary with evaluation metrics
    """
    n_total = 0
    n_clean_correct = 0
    n_success = 0
    total_changes = 0

    for i, (toks, label) in enumerate(zip(tokens_list, labels)):
        # Limit number of examples if specified
        if max_examples is not None and i >= max_examples:
            break

        true_label = int(label)

        # Evaluate clean prediction (with or without sanitization)
        clean_toks = sanitize_tokens(toks, vocab) if use_sanitization else toks
        clean_ids = encode_tokens(clean_toks, vocab, max_len)
        x_clean = torch.tensor(clean_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_clean = attack_obj.model(x_clean)
            pred_clean = int(logits_clean.argmax(dim=1).item())

        n_total += 1
        
        # Only attack examples that are correctly classified
        if pred_clean != true_label:
            continue

        n_clean_correct += 1

        # Generate adversarial example on raw tokens (without sanitization)
        toks_padded = list(toks)
        if len(toks_padded) < max_len:
            toks_padded += ["<pad>"] * (max_len - len(toks_padded))
        else:
            toks_padded = toks_padded[:max_len]

        ids_raw = encode_tokens(toks_padded, vocab, max_len)
        input_ids = torch.tensor(ids_raw, dtype=torch.long)

        # Generate adversarial example
        adv_ids_raw, adv_tokens_raw, success, num_changes = attack_obj.attack_example(
            input_ids, toks_padded, true_label, max_change
        )

        total_changes += num_changes

        # Evaluate defended model on adversarial example
        # Apply sanitization if enabled
        adv_toks_for_model = sanitize_tokens(adv_tokens_raw, vocab) if use_sanitization else adv_tokens_raw
        adv_ids_for_model = encode_tokens(adv_toks_for_model, vocab, max_len)
        x_adv = torch.tensor(adv_ids_for_model, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_adv = attack_obj.model(x_adv)
            pred_adv = int(logits_adv.argmax(dim=1).item())

        # Count successful attacks (misclassification after defense)
        if success and pred_adv != true_label:
            n_success += 1

    # Compute metrics
    if n_clean_correct == 0:
        asr = 0.0
        robust_acc = 0.0
        avg_changes = 0.0
    else:
        asr = n_success / n_clean_correct  # Attack success rate
        robust_acc = (n_clean_correct - n_success) / n_clean_correct  # Robust accuracy
        avg_changes = total_changes / n_clean_correct  # Average changes per example

    return {
        "attack": attack_name,
        "budget": max_change,
        "n_total": n_total,
        "n_clean_correct": n_clean_correct,
        "n_success": n_success,
        "asr": asr,
        "robust_acc": robust_acc,
        "avg_changes": avg_changes,
    }


def main():
    """Main evaluation function."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configuration YAML file")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument(
        "--output_csv",
        default="experiments/logs/results_attacks.csv",
        help="Where to save attack results",
    )
    ap.add_argument(
        "--sanitize",
        action="store_true",
        help="Enable input sanitization defense during evaluation",
    )
    args = ap.parse_args()

    # Load configuration
    cfg = load_cfg(args.config)
    attacks_cfg = cfg.get("attacks", {})

    # Get maximum number of examples to evaluate
    max_examples = attacks_cfg.get("max_eval_examples", None)
    if isinstance(max_examples, str) and max_examples.lower() == "none":
        max_examples = None

    # Load model and vocabulary
    model, V, device = build_model_and_vocab(cfg, args.ckpt)
    print("Using device:", device)

    # Prepare test dataset
    tokens_list, labels = prepare_dataset(cfg)
    print(f"Loaded {len(tokens_list)} test examples.")

    # Get attack configurations
    kw_cfg = attacks_cfg.get("keyword", {})
    ch_cfg = attacks_cfg.get("char", {})

    # Initialize attack objects
    kw_attack = KeywordSubstitutionAttack(
        model=model,
        vocab=V,
        device=device,
        pad_idx=0,
        unk_idx=1,
        max_fraction_changed=kw_cfg.get("max_fraction_changed", 0.2),
        top_k_words=kw_cfg.get("top_k_words", 8),
        max_synonyms=kw_cfg.get("max_synonyms", 20),
        random_seed=cfg["project"]["seed"],
    )

    ch_attack = CharPerturbationAttack(
        model=model,
        vocab=V,
        device=device,
        pad_idx=0,
        unk_idx=1,
        max_char_frac=ch_cfg.get("max_char_frac", 0.15),
        random_seed=cfg["project"]["seed"],
    )

    # Get attack budgets to test
    keyword_budgets = kw_cfg.get("max_changes_list", [1, 2, 3])
    char_budgets = ch_cfg.get("max_edits_list", [1, 2, 3])

    max_len = cfg["data"]["max_len"]
    use_sanitization = bool(args.sanitize)
    
    if use_sanitization:
        print(">>> Input sanitization defense ENABLED")
    else:
        print(">>> Input sanitization defense DISABLED")

    results = []

    # Evaluate keyword substitution attack with different budgets
    for k in keyword_budgets:
        print(f"\nRunning KeywordSubstitutionAttack with budget={k}...")
        res = evaluate_attack_on_dataset(
            attack_name="keyword_substitution",
            attack_obj=kw_attack,
            tokens_list=tokens_list,
            labels=labels,
            vocab=V,
            device=device,
            max_change=int(k),
            max_examples=max_examples,
            max_len=max_len,
            use_sanitization=use_sanitization,
        )
        print(res)
        results.append(res)

    # Evaluate character perturbation attack with different budgets
    for m in char_budgets:
        print(f"\nRunning CharPerturbationAttack with budget={m}...")
        res = evaluate_attack_on_dataset(
            attack_name="char_perturbation",
            attack_obj=ch_attack,
            tokens_list=tokens_list,
            labels=labels,
            vocab=V,
            device=device,
            max_change=int(m),
            max_examples=max_examples,
            max_len=max_len,
            use_sanitization=use_sanitization,
        )
        print(res)
        results.append(res)

    # Save results to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = [
        "attack",
        "budget",
        "n_total",
        "n_clean_correct",
        "n_success",
        "asr",
        "robust_acc",
        "avg_changes",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSaved attack results to {args.output_csv}")


if __name__ == "__main__":
    main()
