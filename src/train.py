import argparse, yaml, os, torch
from torch import nn
from torch.optim import Adam
from src.data import build_imdb_splits, make_loaders
from src.models.cnn_text import TextCNN
from src.utils.metrics import Accumulator

def load_cfg(p): return yaml.safe_load(open(p))
def set_seed(s):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s);
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def get_device(cfg_device: str):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(cfg_device)

def main(args):
    cfg = load_cfg(args.config)
    set_seed(cfg["project"]["seed"])
    (trX,trY),(vaX,vaY),(teX,teY), V, meta = build_imdb_splits(cfg)
    tr_loader, va_loader = make_loaders((trX,trY), (vaX,vaY), cfg)

    device = get_device(cfg["project"]["device"])
    print("Using device:", device)
    model = TextCNN(vocab_size=meta["vocab_size"],
                    emb_dim=cfg["model"]["embedding_dim"],
                    num_classes=cfg["model"]["num_classes"],
                    filter_sizes=cfg["model"]["filter_sizes"],
                    num_filters=cfg["model"]["num_filters"],
                    dropout=cfg["model"]["dropout"],
                    pad_idx=0).to(device)

    opt = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    crit = nn.CrossEntropyLoss()

    best_f1, patience = -1, 0
    os.makedirs(cfg["log"]["dir"], exist_ok=True)
    csv_path = os.path.join(cfg["log"]["dir"], cfg["log"]["csv_name"])
    with open(csv_path, "w") as f: f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1\n")

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        meter = Accumulator()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.long().to(device)
            opt.zero_grad(); logits = model(xb); loss = crit(logits, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            meter.add_batch(logits, yb, loss.item())

        tr_loss, tr_acc = meter.loss(), meter.acc()

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}")

        # val
        model.eval(); meter = Accumulator()
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.long().to(device)
                logits = model(xb); loss = crit(logits, yb)
                meter.add_batch(logits, yb, loss.item())
        va_loss, va_acc, va_f1 = meter.loss(), meter.acc(), meter.f1()
        with open(csv_path, "a") as f: f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{va_loss:.4f},{va_acc:.4f},{va_f1:.4f}\n")

        if va_f1 > best_f1:
            best_f1, patience = va_f1, 0
            torch.save(model.state_dict(), cfg["log"]["ckpt_path"])
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]: break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args())