import argparse
import os
import pickle
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import Dataset, DataLoader

from src.data import read_split, clean_text
from src.vocab import tokenize
from src.models.cnn_text import TextCNN
from src.attacks.text_attacks import KeywordSubstitutionAttack, CharPerturbationAttack
from src.utils.metrics import Accumulator


# ----------------- helpers ----------------- #

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AdvDataset(Dataset):
    """
    Stores tokenized texts and labels. We encode with the vocab inside the training loop
    so we can still access tokens easily for attacks.
    """
    def __init__(self, tokens: List[List[str]], labels: List[int]):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def collate_fn(batch):
    tokens_batch, labels_batch = zip(*batch)
    labels = torch.tensor(labels_batch, dtype=torch.long)
    return list(tokens_batch), labels


def prepare_splits(cfg, vocab) -> Tuple[Dataset, Dataset]:
    root = cfg["data"]["root"]
    val_ratio = cfg["data"]["val_ratio"]
    lower = cfg["data"]["lowercase"]
    rm_br = cfg["data"]["remove_html_breaks"]
    norm = cfg["data"]["normalize_unicode"]

    # full labeled train set
    train_all = read_split(root, "train")  # list[(text, label)]

    # shuffle deterministically
    seed = cfg["project"].get("seed", 1337)
    rnd = random.Random(seed)
    rnd.shuffle(train_all)

    n_total = len(train_all)
    n_train = int((1.0 - val_ratio) * n_total)
    train_split = train_all[:n_train]
    val_split = train_all[n_train:]

    def prep(split):
        texts = [clean_text(x, lower, rm_br, norm) for x, _ in split]
        labels = [y for _, y in split]
        tokens = [tokenize(t) for t in texts]
        return tokens, labels

    tr_tokens, tr_labels = prep(train_split)
    va_tokens, va_labels = prep(val_split)

    tr_ds = AdvDataset(tr_tokens, tr_labels)
    va_ds = AdvDataset(va_tokens, va_labels)
    return tr_ds, va_ds


def encode_batch(tokens_batch: List[List[str]], vocab, max_len: int) -> torch.Tensor:
    ids = [vocab.encode(toks, max_len) for toks in tokens_batch]
    return torch.tensor(ids, dtype=torch.long)


# ----------------- main adversarial training ----------------- #

def main(args):
    cfg = load_cfg(args.config)
    adv_cfg = cfg["adv_train"]

    seed = cfg["project"].get("seed", 1337)
    set_seed(seed)

    # device
    if cfg["project"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["project"]["device"])
    print("Using device:", device)

    # load vocab (must match baseline)
    vocab_path = os.path.join(cfg["log"]["dir"], "vocab.pkl")
    with open(vocab_path, "rb") as f:
        V = pickle.load(f)

    # build datasets
    train_ds, val_ds = prepare_splits(cfg, V)
    bs = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    max_len = cfg["data"]["max_len"]

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

    # load baseline checkpoint
    base_ckpt = adv_cfg["from_ckpt"]
    state = torch.load(base_ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded baseline weights from {base_ckpt}")

    # optimizer / loss
    lr = adv_cfg.get("lr", 5e-4)
    wd = adv_cfg.get("weight_decay", 0.0)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    # attacks for generating adversarial examples
    attacks_list = adv_cfg.get("attacks", ["keyword", "char"])
    max_changes_list = adv_cfg.get("max_changes", [1, 2])

    kw_attack = None
    ch_attack = None
    if "keyword" in attacks_list:
        kw_attack = KeywordSubstitutionAttack(
            model=model,
            vocab=V,
            device=device,
            pad_idx=0,
            unk_idx=1,
            max_fraction_changed=cfg["attacks"]["keyword"]["max_fraction_changed"],
            top_k_words=cfg["attacks"]["keyword"]["top_k_words"],
            max_synonyms=cfg["attacks"]["keyword"]["max_synonyms"],
            random_seed=seed,
        )
    if "char" in attacks_list:
        ch_attack = CharPerturbationAttack(
            model=model,
            vocab=V,
            device=device,
            pad_idx=0,
            unk_idx=1,
            max_char_frac=cfg["attacks"]["char"]["max_char_frac"],
            random_seed=seed,
        )

    adv_ratio = adv_cfg.get("adv_ratio", 0.5)
    epochs = adv_cfg.get("epochs", 3)

    log_dir = cfg["log"]["dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_csv = adv_cfg["log_csv"]

    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1\n")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} (adversarial training) ===")
        model.train()
        if kw_attack is not None:
            kw_attack.refresh_embeddings()   # embeddings changed last epoch

        meter_tr = Accumulator()

        for tokens_batch, yb in train_loader:
            yb = yb.to(device)
            xb = encode_batch(tokens_batch, V, max_len)  # clean ids (on CPU)
            xb_adv = xb.clone()

            batch_size = xb.size(0)
            n_adv = int(adv_ratio * batch_size)
            indices = list(range(batch_size))
            random.shuffle(indices)
            adv_indices = indices[:n_adv]

            for idx in adv_indices:
                ids_i = xb_adv[idx].clone()        # (T,)
                toks_i = tokens_batch[idx]
                # pad / truncate tokens as encode() does
                toks_padded = list(toks_i)
                if len(toks_padded) < max_len:
                    toks_padded += ["<pad>"] * (max_len - len(toks_padded))
                else:
                    toks_padded = toks_padded[:max_len]

                true_label = int(yb[idx].item())
                budget = random.choice(max_changes_list)

                # choose which attack to use
                if kw_attack is not None and ch_attack is not None:
                    use_kw = random.random() < 0.5
                elif kw_attack is not None:
                    use_kw = True
                elif ch_attack is not None:
                    use_kw = False
                else:
                    use_kw = False  # should never happen

                if use_kw:
                    adv_ids, _, _, _ = kw_attack.attack_example(
                        ids_i.cpu(), toks_padded, true_label, max_changes=budget
                    )
                else:
                    adv_ids, _, _, _ = ch_attack.attack_example(
                        ids_i.cpu(), toks_padded, true_label, max_edits=budget
                    )

                xb_adv[idx] = adv_ids

            xb_adv = xb_adv.to(device)

            opt.zero_grad()
            logits = model(xb_adv)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

            meter_tr.add_batch(logits, yb, float(loss.item()))

        tr_loss, tr_acc = meter_tr.loss(), meter_tr.acc()
        print(f"Train: loss={tr_loss:.4f}  acc={tr_acc:.4f}")

        # -------- validation on clean data -------- #
        model.eval()
        meter_va = Accumulator()
        with torch.no_grad():
            for tokens_batch, yb in val_loader:
                yb = yb.to(device)
                xb = encode_batch(tokens_batch, V, max_len).to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                meter_va.add_batch(logits, yb, float(loss.item()))

        va_loss, va_acc, va_f1 = meter_va.loss(), meter_va.acc(), meter_va.f1()
        print(f"Val  : loss={va_loss:.4f}  acc={va_acc:.4f}  f1={va_f1:.4f}")

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{va_loss:.4f},{va_acc:.4f},{va_f1:.4f}\n")

    # save adv-trained checkpoint
    out_ckpt = adv_cfg["out_ckpt"]
    torch.save(model.state_dict(), out_ckpt)
    print(f"\nSaved adversarially trained model to {out_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
