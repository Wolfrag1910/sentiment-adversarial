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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_vocab(cfg, ckpt_path):
    # device
    if cfg["project"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["project"]["device"])

    # vocab saved during training
    vocab_path = os.path.join(cfg["log"]["dir"], "vocab.pkl")
    with open(vocab_path, "rb") as f:
        V = pickle.load(f)

    # model
    model = TextCNN(
        vocab_size=len(V.itos),
        emb_dim=cfg["model"]["embedding_dim"],
        num_classes=cfg["model"]["num_classes"],
        filter_sizes=cfg["model"]["filter_sizes"],
        num_filters=cfg["model"]["num_filters"],
        dropout=cfg["model"]["dropout"],
        pad_idx=0,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, V, device


def prepare_dataset(cfg):
    root = cfg["data"]["root"]
    test_raw = read_split(root, "test")  # list[(text, label)]

    lower = cfg["data"]["lowercase"]
    rm_br = cfg["data"]["remove_html_breaks"]
    norm = cfg["data"]["normalize_unicode"]

    texts = [clean_text(x, lower, rm_br, norm) for x, _ in test_raw]
    labels = [y for _, y in test_raw]
    tokens_list = [tokenize(t) for t in texts]
    return tokens_list, labels


def encode_tokens(tokens: List[str], vocab, max_len: int) -> List[int]:
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
    n_total = 0
    n_clean_correct = 0
    n_success = 0
    total_changes = 0

    for i, (toks, label) in enumerate(zip(tokens_list, labels)):
        if max_examples is not None and i >= max_examples:
            break

        true_label = int(label)

        # ----- clean prediction (with or without sanitization) -----
        clean_toks = sanitize_tokens(toks, vocab) if use_sanitization else toks
        clean_ids = encode_tokens(clean_toks, vocab, max_len)
        x_clean = torch.tensor(clean_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_clean = attack_obj.model(x_clean)
            pred_clean = int(logits_clean.argmax(dim=1).item())

        n_total += 1
        if pred_clean != true_label:
            # only attack originally correct examples
            continue

        n_clean_correct += 1

        # ----- generate adversarial example on *raw* tokens -----
        toks_padded = list(toks)
        if len(toks_padded) < max_len:
            toks_padded += ["<pad>"] * (max_len - len(toks_padded))
        else:
            toks_padded = toks_padded[:max_len]

        ids_raw = encode_tokens(toks_padded, vocab, max_len)
        input_ids = torch.tensor(ids_raw, dtype=torch.long)

        adv_ids_raw, adv_tokens_raw, success, num_changes = attack_obj.attack_example(
            input_ids, toks_padded, true_label, max_change
        )

        total_changes += num_changes

        # ----- run the *defended* model on adversarial example -----
        adv_toks_for_model = sanitize_tokens(adv_tokens_raw, vocab) if use_sanitization else adv_tokens_raw
        adv_ids_for_model = encode_tokens(adv_toks_for_model, vocab, max_len)
        x_adv = torch.tensor(adv_ids_for_model, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_adv = attack_obj.model(x_adv)
            pred_adv = int(logits_adv.argmax(dim=1).item())

        if success and pred_adv != true_label:
            n_success += 1

    if n_clean_correct == 0:
        asr = 0.0
        robust_acc = 0.0
        avg_changes = 0.0
    else:
        asr = n_success / n_clean_correct
        robust_acc = (n_clean_correct - n_success) / n_clean_correct
        avg_changes = total_changes / n_clean_correct

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--output_csv",
        default="experiments/logs/results_attacks.csv",
        help="Where to save attack results.",
    )
    ap.add_argument(
        "--sanitize",
        action="store_true",
        help="Enable input sanitization defense during evaluation.",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    attacks_cfg = cfg.get("attacks", {})

    max_examples = attacks_cfg.get("max_eval_examples", None)
    if isinstance(max_examples, str) and max_examples.lower() == "none":
        max_examples = None

    model, V, device = build_model_and_vocab(cfg, args.ckpt)
    print("Using device:", device)

    tokens_list, labels = prepare_dataset(cfg)
    print(f"Loaded {len(tokens_list)} test examples.")

    kw_cfg = attacks_cfg.get("keyword", {})
    ch_cfg = attacks_cfg.get("char", {})

    # Attack objects share the same model
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

    keyword_budgets = kw_cfg.get("max_changes_list", [1, 2, 3])
    char_budgets = ch_cfg.get("max_edits_list", [1, 2, 3])

    max_len = cfg["data"]["max_len"]
    use_sanitization = bool(args.sanitize)
    if use_sanitization:
        print(">>> Input sanitization defense ENABLED")
    else:
        print(">>> Input sanitization defense DISABLED")

    results = []

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

    # Save to CSV
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
