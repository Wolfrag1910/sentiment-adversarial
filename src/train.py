"""
Baseline training script for Text CNN on IMDB sentiment classification.

Trains a clean CNN model without adversarial examples. This serves as the
baseline for evaluating adversarial robustness.
"""

import argparse
import yaml
import os
import torch
from torch import nn
from torch.optim import Adam
from src.data import build_imdb_splits, make_loaders
from src.models.cnn_text import TextCNN
from src.utils.metrics import Accumulator
import pickle


def load_cfg(p):
    """Load configuration from YAML file."""
    return yaml.safe_load(open(p))


def set_seed(s):
    """
    Set random seeds for reproducibility.
    
    Args:
        s: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    
    # Set CUDA seed if GPU is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device(cfg_device: str):
    """
    Get the device for training (CPU or CUDA).
    
    Args:
        cfg_device: Device string from config ('auto', 'cuda', or 'cpu')
        
    Returns:
        torch.device object
    """
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(cfg_device)


def main(args):
    """Main training loop."""
    # Load configuration
    cfg = load_cfg(args.config)
    
    # Set random seed for reproducibility
    set_seed(cfg["project"]["seed"])
    
    # Build train/val/test splits and vocabulary
    (trX, trY), (vaX, vaY), (teX, teY), V, meta = build_imdb_splits(cfg)
    
    # Save vocabulary for later use in evaluation and adversarial training
    os.makedirs(cfg["log"]["dir"], exist_ok=True)
    vocab_path = os.path.join(cfg["log"]["dir"], "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(V, f)
    
    # Create data loaders
    tr_loader, va_loader = make_loaders((trX, trY), (vaX, vaY), cfg)

    # Set up device (CPU or GPU)
    device = get_device(cfg["project"]["device"])
    print("Using device:", device)
    
    # Initialize model
    model = TextCNN(
        vocab_size=meta["vocab_size"],
        emb_dim=cfg["model"]["embedding_dim"],
        num_classes=cfg["model"]["num_classes"],
        filter_sizes=cfg["model"]["filter_sizes"],
        num_filters=cfg["model"]["num_filters"],
        dropout=cfg["model"]["dropout"],
        pad_idx=0
    ).to(device)

    # Set up optimizer and loss function
    opt = Adam(model.parameters(), lr=cfg["train"]["lr"], 
               weight_decay=cfg["train"]["weight_decay"])
    crit = nn.CrossEntropyLoss()

    # Initialize early stopping variables
    best_f1 = -1
    patience = 0
    
    # Create log directory and CSV file for training metrics
    os.makedirs(cfg["log"]["dir"], exist_ok=True)
    csv_path = os.path.join(cfg["log"]["dir"], cfg["log"]["csv_name"])
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1\n")

    # Training loop
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # Training phase
        model.train()
        meter = Accumulator()
        
        for xb, yb in tr_loader:
            # Move data to device
            xb = xb.to(device)
            yb = yb.long().to(device)
            
            # Forward pass
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            
            # Update weights
            opt.step()
            
            # Accumulate metrics
            meter.add_batch(logits, yb, loss.item())

        # Compute training metrics
        tr_loss, tr_acc = meter.loss(), meter.acc()
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}")

        # Validation phase
        model.eval()
        meter = Accumulator()
        
        with torch.no_grad():
            for xb, yb in va_loader:
                # Move data to device
                xb = xb.to(device)
                yb = yb.long().to(device)
                
                # Forward pass
                logits = model(xb)
                loss = crit(logits, yb)
                
                # Accumulate metrics
                meter.add_batch(logits, yb, loss.item())
        
        # Compute validation metrics
        va_loss, va_acc, va_f1 = meter.loss(), meter.acc(), meter.f1()
        
        # Log metrics to CSV
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{va_loss:.4f},{va_acc:.4f},{va_f1:.4f}\n")

        # Early stopping based on validation F1 score
        if va_f1 > best_f1:
            best_f1 = va_f1
            patience = 0
            # Save best model checkpoint
            torch.save(model.state_dict(), cfg["log"]["ckpt_path"])
        else:
            patience += 1
            # Stop training if no improvement for patience epochs
            if patience >= cfg["train"]["early_stop_patience"]:
                break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configuration YAML file")
    main(ap.parse_args())
