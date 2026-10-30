"""
Pairwise reading-order model: a small MLP that looks at two boxes (a, b) on a page and predicts
"is a read before b?". A page order is then built from all the pair answers.

  features of one box:  x1, y1, x2, y2, centre x, centre y, width, height, class (one-hot)
  features of a pair:   box a, box b, their offset (dx, dy), how much they overlap in x and in y,
                        and (with --xycut-feature) whether XY-cut puts a before b
  page order:           each box gets the sum of its "before" probabilities against every other box;
                        sort by that score (highest = read first)

Trained on the train pages, early stopping on the val pages, scored on the test pages.
Because there are only ~74 training pages, the model is trained with several seeds and the
scores are averaged.

    python train_pairwise.py --name P1_mlp
    python train_pairwise.py --name P2_mlp_xycut --xycut-feature

Each run saves to runs_order/<name>/: model.pt (all seeds), loss_curve.png, history.csv (first seed), predictions.csv,
and adds one row to runs_order/summary.csv.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from reading_order_utils import create_pair_mlp, load_split_pages, mlp_order, order_scores, page_pairs, save_results

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--xycut-feature", action="store_true", help="add 'XY-cut puts a before b' as a feature")
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--l2", type=float, default=1e-4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--seeds", type=int, default=5)
args = parser.parse_args()

OUT_DIR = os.path.join("runs_order", args.name)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Data: one example per ordered pair of boxes on a page
# ============================================================
split_pages = load_split_pages()


def split_tensors(split):
    X, y = [], []
    for p in split_pages[split]:
        Xp, yp, _ = page_pairs(p["boxes"], args.xycut_feature)
        X += Xp
        y += yp
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


X_train, y_train = split_tensors("train")
X_val, y_val = split_tensors("val")
print(f"pages  train {len(split_pages['train'])}  val {len(split_pages['val'])}  test {len(split_pages['test'])}")
print(f"pairs  train {len(y_train)}  val {len(y_val)}   features per pair {X_train.shape[1]}")

# ============================================================
# 2. Model
# ============================================================


def train_one(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = create_pair_mlp(X_train.shape[1], args.hidden, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)  # weight_decay = L2
    loss_fn = nn.BCEWithLogitsLoss()
    history, best_val, best_state, wait = [], float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(y_train))
        total = 0.0
        for k in range(0, len(perm), args.batch_size):
            idx = perm[k:k + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]).squeeze(1), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val).squeeze(1), y_val).item()
            train_acc = ((model(X_train).squeeze(1) > 0).float() == y_train).float().mean().item()
            val_acc = ((model(X_val).squeeze(1) > 0).float() == y_val).float().mean().item()
        history.append(dict(epoch=epoch, train_loss=total / len(y_train), val_loss=val_loss,
                            train_pair_acc=train_acc, val_pair_acc=val_acc))

        if val_loss < best_val:   # early stopping on val loss
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def evaluate(model, split):
    return [order_scores(p["boxes"], mlp_order(model, p["boxes"], args.xycut_feature)) for p in split_pages[split]]


# ============================================================
# 3. Train with several seeds, score val and test
# ============================================================
all_test, all_val, histories, states = [], [], [], []
for seed in range(args.seeds):
    model, history = train_one(seed)
    histories.append(history)
    all_val.append(evaluate(model, "val"))
    all_test.append(evaluate(model, "test"))
    states.append(model.state_dict())
    tau = lambda rows: sum(r["kendall_tau"] for r in rows) / len(rows)
    print(f"seed {seed}: stopped at epoch {len(history)}, val tau {tau(all_val[-1]):.3f}, test tau {tau(all_test[-1]):.3f}")

# all seeds are kept: the reading pipeline (read_page.py) averages their answers (an ensemble)
torch.save({"states": states, "n_features": X_train.shape[1], "hidden": args.hidden,
            "dropout": args.dropout, "xycut_feature": args.xycut_feature}, os.path.join(OUT_DIR, "model.pt"))

setting = f"hidden {args.hidden}, dropout {args.dropout}, l2 {args.l2}" + (", xy-cut feature" if args.xycut_feature else "")
save_results(OUT_DIR, args.name, "pairwise MLP", setting, split_pages["test"], all_test, all_val, histories)
