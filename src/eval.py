"""
Model evaluation script for clean test data.

Evaluates a trained model on the IMDB test set without adversarial perturbations.
"""

import argparse
import os
import pickle
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import read_split, clean_text, TextDataset
from src.vocab import tokenize
from src.models.cnn_text import TextCNN
from src.utils.metrics import Accumulator


def load_cfg(p):
    """Load configuration from YAML file."""
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args):
    """Main evaluation function."""
    # Load configuration
    cfg = load_cfg(args.config)

    # Set up device (CPU or GPU)
    if cfg["project"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["project"]["device"])
    print("Using device:", device)

    # Load vocabulary saved during training
    vocab_path = os.path.join(cfg["log"]["dir"], "vocab.pkl")
    with open(vocab_path, "rb") as f:
        V = pickle.load(f)

    # Load raw test data
    test_raw = read_split(cfg["data"]["root"], "test")

    # Clean and tokenize with same settings as training
    lower = cfg["data"]["lowercase"]
    rm_br = cfg["data"]["remove_html_breaks"]
    norm = cfg["data"]["normalize_unicode"]
    max_len = cfg["data"]["max_len"]

    clean_texts = [clean_text(x, lower, rm_br, norm) for x, _ in test_raw]
    toks = [tokenize(t) for t in clean_texts]
    labels = [y for _, y in test_raw]

    # Encode using loaded vocabulary
    encoded = [V.encode(t, max_len) for t in toks]

    # Build DataLoader
    test_ds = TextDataset(encoded, labels)
    bs = cfg["data"]["batch_size"]
    test_loader = DataLoader(
        test_ds, 
        batch_size=bs, 
        shuffle=False,
        num_workers=cfg["data"]["num_workers"]
    )

    # Build model with vocabulary size from loaded vocab
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
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Evaluate on test set
    meter = Accumulator()
    crit = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            meter.add_batch(logits, yb, float(loss.item()))

    # Print results
    print(f"TEST  loss={meter.loss():.4f}  acc={meter.acc():.4f}  f1={meter.f1():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configuration YAML file")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    main(ap.parse_args())
