import argparse, yaml, torch, os
from src.data import build_imdb_splits
from src.models.cnn_text import TextCNN
from src.utils.metrics import Accumulator
from torch import nn

def load_cfg(p): return yaml.safe_load(open(p))

def main(args):
    cfg = load_cfg(args.config)
    (_, _), (_, _), (teX, teY), V, meta = build_imdb_splits(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(meta["vocab_size"], cfg["model"]["embedding_dim"], cfg["model"]["num_classes"],
                    cfg["model"]["filter_sizes"], cfg["model"]["num_filters"], cfg["model"]["dropout"], pad_idx=0).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval(); meter = Accumulator(); crit = nn.CrossEntropyLoss()

    import math
    bs = cfg["data"]["batch_size"]
    for i in range(0, len(teX), bs):
        xb = torch.tensor(teX[i:i+bs]).to(device)
        yb = torch.tensor(teY[i:i+bs]).long().to(device)
        with torch.no_grad():
            logits = model(xb); loss = crit(logits, yb)
        meter.add_batch(logits, yb, loss.item())

    print(f"TEST  loss={meter.loss():.4f}  acc={meter.acc():.4f}  f1={meter.f1():.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    main(ap.parse_args())
