import argparse
import csv
import os
import pickle
from typing import Dict, Any

import torch
import yaml

from src.data import read_split, clean_text
from src.vocab import tokenize
from src.models.cnn_text import TextCNN
from src.attacks.text_attacks import KeywordSubstitutionAttack, CharPerturbationAttack


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


def prepare_dataset(cfg, V):
    root = cfg["data"]["root"]
    test_raw = read_split(root, "test")  # list[(text, label)]

    lower = cfg["data"]["lowercase"]
    rm_br = cfg["data"]["remove_html_breaks"]
    norm = cfg["data"]["normalize_unicode"]
    max_len = cfg["data"]["max_len"]

    texts = [clean_text(x, lower, rm_br, norm) for x, _ in test_raw]
    labels = [y for _, y in test_raw]
    tokens_list = [tokenize(t) for t in texts]
    ids_list = [V.encode(toks, max_len) for toks in tokens_list]
    return ids_list, tokens_list, labels


def evaluate_attack_on_dataset(
    attack_name: str,
    attack_obj,
    ids_list,
    tokens_list,
    labels,
    device,
    max_change: int,
    max_examples: int | None,
):
    n_total = 0
    n_clean_correct = 0
    n_success = 0
    total_changes = 0

    for i, (ids, toks, label) in enumerate(zip(ids_list, tokens_list, labels)):
        if max_examples is not None and i >= max_examples:
            break

        input_ids = torch.tensor(ids, dtype=torch.long).to(device)
        true_label = int(label)

        # clean prediction
        with torch.no_grad():
            logits = attack_obj.model(input_ids.unsqueeze(0))
            preds = logits.argmax(dim=1)
        pred_label = int(preds.item())

        n_total += 1
        if pred_label != true_label:
            # only attack originally correct examples when computing ASR
            continue

        n_clean_correct += 1

        # pad tokens up to length of ids (for safe indexing)
        toks_padded = list(toks)
        if len(toks_padded) < len(ids):
            toks_padded += ["<pad>"] * (len(ids) - len(toks_padded))

        # attack (attack_example expects CPU tensor)
        adv_ids, adv_tokens, success, num_changes = attack_obj.attack_example(
            input_ids.cpu(), toks_padded, true_label, max_change
        )

        total_changes += num_changes

        # final prediction
        with torch.no_grad():
            logits_adv = attack_obj.model(adv_ids.unsqueeze(0).to(device))
            pred_adv = logits_adv.argmax(dim=1).item()

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
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    attacks_cfg = cfg.get("attacks", {})

    max_examples = attacks_cfg.get("max_eval_examples", None)
    if isinstance(max_examples, str) and max_examples.lower() == "none":
        max_examples = None

    model, V, device = build_model_and_vocab(cfg, args.ckpt)
    print("Using device:", device)

    ids_list, tokens_list, labels = prepare_dataset(cfg, V)
    print(f"Loaded {len(ids_list)} test examples.")

    kw_cfg = attacks_cfg.get("keyword", {})
    ch_cfg = attacks_cfg.get("char", {})

    # Attack objects
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

    results = []

    for k in keyword_budgets:
        print(f"\nRunning KeywordSubstitutionAttack with budget={k}...")
        res = evaluate_attack_on_dataset(
            attack_name="keyword_substitution",
            attack_obj=kw_attack,
            ids_list=ids_list,
            tokens_list=tokens_list,
            labels=labels,
            device=device,
            max_change=int(k),
            max_examples=max_examples,
        )
        print(res)
        results.append(res)

    for m in char_budgets:
        print(f"\nRunning CharPerturbationAttack with budget={m}...")
        res = evaluate_attack_on_dataset(
            attack_name="char_perturbation",
            attack_obj=ch_attack,
            ids_list=ids_list,
            tokens_list=tokens_list,
            labels=labels,
            device=device,
            max_change=int(m),
            max_examples=max_examples,
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
