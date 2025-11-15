import argparse
import os
import random
from collections import Counter

import numpy as np
import yaml

from src.data import read_split, clean_text
from src.vocab import tokenize


def describe_split(name, examples, cfg, save_hist=True, show_examples=True):
    """
    examples: list[(raw_text, label)]
    """
    lower = cfg["data"]["lowercase"]
    rm_br = cfg["data"]["remove_html_breaks"]
    norm = cfg["data"]["normalize_unicode"]

    # Clean + tokenize
    clean_texts = [clean_text(t, lower, rm_br, norm) for t, _ in examples]
    labels = [y for _, y in examples]
    tokens = [tokenize(t) for t in clean_texts]
    lengths = np.array([len(toks) for toks in tokens], dtype=np.int32)

    # Label distribution
    label_counts = Counter(labels)
    n_total = len(labels)
    n_pos = label_counts.get(1, 0)
    n_neg = label_counts.get(0, 0)

    print(f"\n===== {name.upper()} SPLIT =====")
    print(f"Total examples: {n_total}")
    print(
        f"  Positive: {n_pos} ({n_pos / n_total:.1%}) | "
        f"Negative: {n_neg} ({n_neg / n_total:.1%})"
    )

    # Length statistics
    print("\nReview length (in tokens):")
    print(f"  min    : {lengths.min()}")
    print(f"  max    : {lengths.max()}")
    print(f"  mean   : {lengths.mean():.1f}")
    print(f"  median : {np.median(lengths):.1f}")
    print(f"  95th % : {np.percentile(lengths, 95):.1f}")
    print(f"  99th % : {np.percentile(lengths, 99):.1f}")

    # Optional histogram for slides
    if save_hist:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not installed; skipping histogram)")
        else:
            os.makedirs("figures", exist_ok=True)
            plt.figure()
            plt.hist(lengths, bins=50)
            plt.xlabel("Review length (tokens)")
            plt.ylabel("Count")
            plt.title(f"{name} review length distribution")
            plt.tight_layout()
            out_path = os.path.join("figures", f"{name.lower()}_length_hist.png")
            plt.savefig(out_path)
            plt.close()
            print(f"  Saved length histogram to {out_path}")

    # A few example reviews for the presentation
    if show_examples:
        def print_examples(label, label_name, k=2, max_chars=400):
            idxs = [i for i, y in enumerate(labels) if y == label][:k]
            print(f"\n{label_name} examples (showing {len(idxs)}):")
            for j, i in enumerate(idxs, start=1):
                text = clean_texts[i].replace("\n", " ")
                if len(text) > max_chars:
                    text = text[:max_chars] + "..."
                print(f"--- {label_name} #{j} ---")
                print(text)
                print()

        print_examples(1, "Positive")
        print_examples(0, "Negative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="experiments/configs/imdb_cnn.yaml",
        help="Path to YAML config (same as used for training).",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg["data"]["root"]
    val_ratio = cfg["data"]["val_ratio"]
    seed = cfg["project"].get("seed", 1337)

    # Use same seed & splitting logic as training, so stats match your model.
    random.seed(seed)
    train_all = read_split(root, "train")
    test_all = read_split(root, "test")

    n_train = int((1.0 - val_ratio) * len(train_all))
    train_split = train_all[:n_train]
    val_split = train_all[n_train:]

    # Describe each split
    describe_split("train", train_split, cfg, save_hist=True, show_examples=True)
    describe_split("val",   val_split,   cfg, save_hist=True, show_examples=False)
    describe_split("test",  test_all,    cfg, save_hist=True, show_examples=False)


if __name__ == "__main__":
    main()
