"""
Transformer reading-order model: looks at ALL boxes of a page at once (self-attention), gives each
box one score, and the page is read in order of score (highest first).

  input per box:   x1, y1, x2, y2, centre x, centre y, width, height, class (one-hot)
                   no position encoding: boxes are a set, their place on the page is in the features
  model:           linear embedding -> transformer encoder (self-attention between all boxes) -> score
  loss:            for every pair (a, b): BCE on score(a) - score(b), target 1 if a is read before b
                   (same pair loss as the MLP, so the two are comparable)

Transfer learning with XY-cut (optional): there are only 74 labeled training pages, but ~4,000
IndicDLP train pages with boxes and no reading order. XY-cut gives each of them an approximate order
("pseudo-labels"). The model is first pre-trained on those, then fine-tuned on your labels.

    python train_transformer.py --name R1_scratch
    python train_transformer.py --name R2_pretrain_finetune --pretrain-epochs 20
    python train_transformer.py --name R3_pretrain_only --pretrain-epochs 20 --no-finetune
    python train_transformer.py --name R4_pretrain_head --pretrain-epochs 20 --freeze-encoder

Each run saves to runs_order/<name>/: loss_curve.png, history.csv (first seed), predictions.csv,
and adds one row to runs_order/summary.csv. The pre-trained weights are saved once in
runs_order/pretrain_xycut_e<N>.pt and reused by later runs.
"""

import argparse
import csv
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn

from reading_order_utils import (DATA_DIR, box_features, is_read, load_gt_boxes, load_split_pages,
                                 order_scores, save_results, xy_cut)

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--d-model", type=int, default=64)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--l2", type=float, default=1e-4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--pretrain-epochs", type=int, default=0, help="pre-train on XY-cut pseudo-labels first")
parser.add_argument("--no-finetune", action="store_true", help="only pre-train (the model just imitates XY-cut)")
parser.add_argument("--freeze-encoder", action="store_true", help="fine-tune only the score layer (like a linear probe)")
args = parser.parse_args()

OUT_DIR = os.path.join("runs_order", args.name)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Data: one example = one page (a set of boxes with their order)
# ============================================================
split_pages = load_split_pages()


def make_batch(pages):
    """Pad pages to the same number of boxes. X: features, O: order position, M: True for a real box."""
    n = max(len(p["boxes"]) for p in pages)
    X = torch.zeros(len(pages), n, len(box_features(pages[0]["boxes"][0])))
    O = torch.zeros(len(pages), n)
    M = torch.zeros(len(pages), n, dtype=torch.bool)
    for i, p in enumerate(pages):
        for j, b in enumerate(p["boxes"]):
            X[i, j] = torch.tensor(box_features(b))
            O[i, j] = b["order"]
            M[i, j] = True
    return X, O, M


def pseudo_pages():
    """IndicDLP train pages (not labeled by you), ordered by XY-cut."""
    pages = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "labels", "train", "*.txt"))):
        page_id = "train/" + os.path.basename(path).replace(".txt", ".png")
        boxes = [b for b in load_gt_boxes(page_id) if is_read(b)]
        if 2 <= len(boxes) <= 40:
            for k, b in enumerate(xy_cut(boxes)):
                b["order"] = k + 1
            pages.append({"page": page_id, "boxes": boxes})
    return pages

# ============================================================
# 2. Model
# ============================================================


class OrderTransformer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.embed = nn.Linear(n_features, args.d_model)
        layer = nn.TransformerEncoderLayer(args.d_model, args.heads, dim_feedforward=2 * args.d_model,
                                           dropout=args.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, args.layers, enable_nested_tensor=False)
        self.score = nn.Linear(args.d_model, 1)

    def forward(self, X, M):
        h = self.encoder(self.embed(X), src_key_padding_mask=~M)   # padded boxes are ignored
        return self.score(h).squeeze(-1)                           # one score per box


loss_fn = nn.BCEWithLogitsLoss(reduction="sum")


def pair_loss(scores, O, M):
    """BCE over every pair (a, b) of real boxes: score(a) - score(b) should be > 0 if a comes first.
    Returns summed loss, number of pairs, number of pairs in the right order."""
    diff = scores[:, :, None] - scores[:, None, :]
    target = (O[:, :, None] < O[:, None, :]).float()
    eye = torch.eye(M.shape[1], dtype=torch.bool)[None]
    valid = M[:, :, None] & M[:, None, :] & ~eye
    loss = loss_fn(diff[valid], target[valid])
    correct = ((diff[valid] > 0).float() == target[valid]).sum().item()
    return loss, valid.sum().item(), correct


def run_epoch(model, pages, optimizer=None):
    """One pass over the pages. Trains if an optimizer is given. Returns mean pair loss and pair accuracy."""
    model.train(optimizer is not None)
    if optimizer:
        pages = random.sample(pages, len(pages))
    total_loss = total_pairs = total_correct = 0
    with torch.set_grad_enabled(optimizer is not None):
        for k in range(0, len(pages), args.batch_size):
            X, O, M = make_batch(pages[k:k + args.batch_size])
            loss, n_pairs, correct = pair_loss(model(X, M), O, M)
            if optimizer:
                optimizer.zero_grad()
                (loss / n_pairs).backward()
                optimizer.step()
            total_loss += loss.item()
            total_pairs += n_pairs
            total_correct += correct
    return total_loss / total_pairs, total_correct / total_pairs


def fit(model, train_pages, epochs, patience, lr):
    """Adam + L2 (weight_decay), early stopping on val loss (your labeled val pages)."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.l2)
    history, best_val, best_state, wait = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_pages, optimizer)
        val_loss, val_acc = run_epoch(model, split_pages["val"])
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_pair_acc=train_acc, val_pair_acc=val_acc))
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return history


def predict_order(model, boxes):
    model.eval()
    X, _, M = make_batch([{"boxes": boxes}])
    with torch.no_grad():
        s = model(X, M)[0].numpy()
    return [boxes[i] for i in np.argsort(-s, kind="stable")]


def evaluate(model, split):
    return [order_scores(p["boxes"], predict_order(model, p["boxes"])) for p in split_pages[split]]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


n_features = len(box_features(split_pages["train"][0]["boxes"][0]))
print(f"pages  train {len(split_pages['train'])}  val {len(split_pages['val'])}  test {len(split_pages['test'])}")

# ============================================================
# 3. Pre-training on XY-cut pseudo-labels (done once, then reused)
# ============================================================
pretrain_path = os.path.join("runs_order", f"pretrain_xycut_e{args.pretrain_epochs}.pt")
pretrain_history = None
if args.pretrain_epochs > 0:
    if os.path.exists(pretrain_path):
        print(f"using pre-trained weights from {pretrain_path}")
        saved = torch.load(pretrain_path)
        pretrained_state, pretrain_history = saved["state"], saved["history"]
    else:
        pages = pseudo_pages()
        print(f"pre-training on {len(pages)} IndicDLP train pages ordered by XY-cut ...")
        set_seed(0)
        model = OrderTransformer(n_features)
        old_batch, args.batch_size = args.batch_size, 64
        pretrain_history = fit(model, pages, args.pretrain_epochs, patience=args.pretrain_epochs, lr=1e-3)
        args.batch_size = old_batch
        for r in pretrain_history:
            print(f"  pre-train epoch {r['epoch']}: train loss {r['train_loss']:.4f}, val loss {r['val_loss']:.4f}, "
                  f"val pair acc {r['val_pair_acc']:.3f}")
        pretrained_state = model.state_dict()
        torch.save({"state": pretrained_state, "history": pretrain_history}, pretrain_path)
    with open(os.path.join(OUT_DIR, "pretrain_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pretrain_history[0].keys()))
        w.writeheader()
        w.writerows(pretrain_history)

# ============================================================
# 4. Fine-tune on your labels with several seeds, score val and test
# ============================================================
all_test, all_val, histories = [], [], []
seeds = 1 if args.no_finetune else args.seeds
for seed in range(seeds):
    set_seed(seed)
    model = OrderTransformer(n_features)
    if args.pretrain_epochs > 0:
        model.load_state_dict(pretrained_state)
    if args.freeze_encoder:   # only the final score layer learns from your labels
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("score")
    if args.no_finetune:
        history = pretrain_history
    else:
        lr = args.lr if args.pretrain_epochs == 0 else args.lr / 3   # gentler steps on pre-trained weights
        history = fit(model, split_pages["train"], args.epochs, args.patience, lr)
    histories.append(history)
    all_val.append(evaluate(model, "val"))
    all_test.append(evaluate(model, "test"))
    tau = lambda rows: sum(r["kendall_tau"] for r in rows) / len(rows)
    print(f"seed {seed}: stopped at epoch {len(history)}, val tau {tau(all_val[-1]):.3f}, test tau {tau(all_test[-1]):.3f}")

setting = (f"d {args.d_model}, layers {args.layers}, heads {args.heads}, dropout {args.dropout}, l2 {args.l2}"
           + (f", pre-train {args.pretrain_epochs} ep on XY-cut" if args.pretrain_epochs else ", from scratch")
           + (", no fine-tune" if args.no_finetune else "")
           + (", encoder frozen" if args.freeze_encoder else ""))
save_results(OUT_DIR, args.name, "transformer", setting, split_pages["test"], all_test, all_val, histories)
