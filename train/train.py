import argparse
import gc
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import InversionDataset
from utils.earlystop import EarlyStopping
from utils.network import Classifier


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train the X-AVDT deepfake detector.")
    parser.add_argument("--data_dir", type=str, required=True, help="Feature data root directory. ex) ./MMDF_pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_epochs", type=int, default=10)  # Paper setting: 2 epochs
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--earlystop", action="store_true")
    parser.add_argument("--earlystop_epoch", type=int, default=5)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--loss_freq", type=int, default=50)
    parser.add_argument("--loader_workers", type=int, default=4)

    parser.add_argument("--output_root", type=str, default="results")
    parser.add_argument("--save_dir", type=str, default="x_avdt")
    parser.add_argument("--norm", type=str, choices=["batch", "instance", "layer"], default="batch")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--alpha", default=0.3, type=float)
    parser.add_argument("--loss_name", default="TripletMarginLoss", type=str)
    parser.add_argument("--embedding_size", default=1024, type=int)
    parser.add_argument("--pos_margin", default=0.3, type=float)
    parser.add_argument("--neg_margin", default=1.0, type=float)
    parser.add_argument("--tau", default=0.5, type=float)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--use_miner", action="store_true")
    parser.add_argument("--memory_size", default=1024, type=int)

    return parser.parse_args()


def train(args):
    from utils.losses import CombinedLoss

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = InversionDataset(data_dir=args.data_dir, split="train")
    val_ds = InversionDataset(data_dir=args.data_dir, split="val")

    loader_kwargs = {"num_workers": args.loader_workers, "worker_init_fn": seed_worker, "pin_memory": torch.cuda.is_available()}
    if args.loader_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    print(f"len(train_ds)={len(train_ds)}, len(val_ds)={len(val_ds)}")

    args.save_dir = os.path.join(args.output_root, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    model = Classifier(norm_layer=args.norm).to(device)

    resume_epoch = 0
    if args.resume:
        match = re.search(r"model_(\d+)\.pt", args.resume)
        if match:
            resume_epoch = int(match.group(1)) + 1

        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state, strict=True)

        print(f"Resumed from {args.resume}, start epoch {resume_epoch}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)

    early_stopping = None
    if args.earlystop:
        early_stopping = EarlyStopping(patience=args.earlystop_epoch, verbose=True, delta=0, path=os.path.join(args.save_dir, "model_best.pt"))

    contrastive_criterion = CombinedLoss(loss_name=args.loss_name, embedding_size=args.embedding_size, pos_margin=args.pos_margin, neg_margin=args.neg_margin,
                            tau=args.tau, memory_size=args.memory_size, use_miner=args.use_miner, num_classes=args.num_classes)

    best_val_acc = -float("inf")
    step = 0

    for epoch in range(resume_epoch, args.total_epochs):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for x, attn, y, _ in tqdm(train_loader, desc=f"[Train {epoch}]"):
            x = x.to(device, dtype=torch.float)
            attn = attn.to(device, dtype=torch.float)
            y = torch.as_tensor(y, device=device)

            out, embed = model(x, attn)

            # 2-class softmax head (matches the released checkpoint's fc=(2,1024)).
            ce = F.cross_entropy(out, y.long())

            embed = F.normalize(embed, dim=1)
            contrastive = contrastive_criterion(embed, y.long())

            loss = (1 - args.alpha) * ce + args.alpha * contrastive

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1).long()

                batch_correct = (preds == y.long()).sum().item()
                batch_count = y.numel()
                batch_acc = batch_correct / max(1, batch_count)

                train_loss_sum += float(loss.detach().cpu()) * batch_count
                train_correct += batch_correct
                train_count += batch_count

            if step % max(1, args.loss_freq) == 0:
                print(
                    f"[Epoch {epoch}][Step {step}] "
                    f"loss={loss.item():.4f} "
                    f"ce={ce.item():.4f} "
                    f"contrastive={contrastive.item():.4f} "
                    f"batch_acc={batch_acc:.4f}"
                )

            step += 1

        train_loss = train_loss_sum / max(1, train_count)
        train_acc = train_correct / max(1, train_count)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for x, attn, y, _ in tqdm(val_loader, desc=f"[Val   {epoch}]"):
                x = x.to(device, dtype=torch.float)
                attn = attn.to(device, dtype=torch.float)
                y = torch.as_tensor(y, device=device)

                out, embed = model(x, attn)

                ce = F.cross_entropy(out, y.long())

                embed = F.normalize(embed, dim=1)
                contrastive = contrastive_criterion(embed, y.long())

                loss = (1 - args.alpha) * ce + args.alpha * contrastive

                probs = torch.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1).long()

                batch_count = y.numel()
                val_loss_sum += float(loss.detach().cpu()) * batch_count
                val_correct += (preds == y.long()).sum().item()
                val_count += batch_count

        val_loss = val_loss_sum / max(1, val_count)
        val_acc = val_correct / max(1, val_count)

        if early_stopping:
            early_stopping(val_acc, model)
            early_stopping.adjust_learning_rate(optimizer, min_lr=args.min_lr)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch} (val acc {val_acc:.4f})")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))
                break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))

        print(f"[Epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{epoch:04d}.pt"))
        gc.collect()


if __name__ == "__main__":
    train(parse_args())