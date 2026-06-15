import argparse
import collections
import json
import os
import re
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import InversionDataset
from utils.network import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an X-AVDT checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True,  help="Feature data root directory. ex) ./MMDF_pt")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--loader_workers", type=int, default=4)
    parser.add_argument("--norm", type=str, choices=["batch", "instance", "layer"], default="batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_model_id", action="store_true", help="Report metrics per generator/model id.")
    parser.add_argument("--bn_recal", type=int, default=0, help="Run BN recalibration with N mini-batches before eval.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output JSON path.")

    return parser.parse_args()


def compute_eer_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(thresholds[idx])


def metrics_block(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    out = {}

    try:
        out["AUROC"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["AUROC"] = None

    try:
        out["AP"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["AP"] = None

    out[f"Accuracy@{threshold:.2f}"] = float(accuracy_score(y_true, y_pred))
    out["Confusion Matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    out["Classification Report"] = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)

    try:
        eer_threshold = compute_eer_threshold(y_true, y_score)
        out["EER_threshold"] = eer_threshold
        out["Acc@EER"] = float(accuracy_score(y_true, (y_score >= eer_threshold).astype(int)))
    except Exception:
        out["EER_threshold"] = None
        out["Acc@EER"] = None

    return out


def fmt4(value):
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint:", ckpt_path)

    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    new_state = collections.OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        new_state[key] = value

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    if missing:
        print("[Warn] missing keys:", missing)
    if unexpected:
        print("[Warn] unexpected keys:", unexpected)


@torch.no_grad()
def bn_recalibrate(model, loader, device, steps):
    if steps <= 0:
        return

    def disable_dropout(module):
        if module.__class__.__name__.startswith("Dropout"):
            module.eval()

    model.apply(disable_dropout)
    model.train()

    for idx, (x, attn, _, _) in enumerate(loader):
        if idx >= steps:
            break

        x = x.to(device, dtype=torch.float)
        attn = attn.to(device, dtype=torch.float)

        model(x, attn)

    model.eval()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(norm_layer=args.norm).to(device)

    load_model(model, args.ckpt)
    model.eval()

    dataset = InversionDataset(data_dir=args.data_dir, split="test")

    if len(dataset) == 0:
        raise RuntimeError(
            f"No test samples found under: {args.data_dir}/test\n"
            f"Expected:\n"
            f"  {args.data_dir}/test/real/...\n"
            f"  {args.data_dir}/test/fake/..."
        )

    loader_kwargs = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.loader_workers, "pin_memory": torch.cuda.is_available()}

    if args.loader_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    loader = DataLoader(dataset, **loader_kwargs)

    if args.bn_recal > 0 and args.norm == "batch":
        bn_recalibrate(model, loader, device, steps=args.bn_recal)

    scores, labels, model_ids = [], [], []

    with torch.inference_mode():
        for x, attn, y, model_id in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device, dtype=torch.float)
            attn = attn.to(device, dtype=torch.float)

            out, _ = model(x, attn)

            # 2-class softmax head: P(fake) = softmax(logits)[:, 1]
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

            scores.extend(prob.tolist())
            labels.extend(y.cpu().numpy().tolist())

            if isinstance(model_id, (list, tuple)):
                model_ids.extend([str(item) for item in model_id])
            else:
                model_ids.append(str(model_id))

    result = {"overall": metrics_block(labels, scores, threshold=0.5)}

    acc_key = next((key for key in result["overall"] if key.startswith("Accuracy@")), None)

    if acc_key is not None:
        result["overall"]["Accuracy"] = float(result["overall"][acc_key])

    summary_overall = (
        f"overall AUROC: {fmt4(result['overall']['AUROC'])}, "
        f"AP: {fmt4(result['overall']['AP'])}, "
        f"Acc: {fmt4(result['overall'].get('Accuracy'))}"
    )

    per_model_summary = {}

    if args.per_model_id:
        scores_by_model = collections.defaultdict(list)
        labels_by_model = collections.defaultdict(list)

        for score, label, model_id in zip(scores, labels, model_ids):
            scores_by_model[model_id].append(score)
            labels_by_model[model_id].append(label)

        result["per_model_id"] = {
            key: metrics_block(labels_by_model[key], scores_by_model[key],  threshold=0.5)
            for key in sorted(labels_by_model.keys())
        }

        for model_id, metrics in result["per_model_id"].items():
            acc_key_model = next((key for key in metrics if key.startswith("Accuracy@")), None)

            if acc_key_model:
                metrics["Accuracy"] = float(metrics[acc_key_model])

            per_model_summary[model_id] = (
                f"{model_id} AUROC: {fmt4(metrics.get('AUROC'))}, "
                f"AP: {fmt4(metrics.get('AP'))}, "
                f"Acc: {fmt4(metrics.get('Accuracy'))}"
            )

    print("==== Overall (thr=0.5) ====")
    print(summary_overall)

    if result["overall"].get("Acc@EER") is not None:
        print(
            f"Acc@EER: {fmt4(result['overall']['Acc@EER'])} "
            f"(EER_threshold={fmt4(result['overall']['EER_threshold'])})"
        )

    if args.per_model_id and "per_model_id" in result:
        print("\n==== Per-Model (AUROC | AP | Acc@EER | EER_thr) ====")

        for model_id in sorted(result["per_model_id"].keys()):
            metrics = result["per_model_id"][model_id]

            print(
                f"[{model_id}] "
                f"AUROC: {fmt4(metrics.get('AUROC'))} | "
                f"AP: {fmt4(metrics.get('AP'))} | "
                f"Acc@EER: {fmt4(metrics.get('Acc@EER'))} | "
                f"EER_thr: {fmt4(metrics.get('EER_threshold'))}"
            )

    save_json = args.save_json or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "score_overall.json")

    epoch = None
    match = re.search(r"model_(\d+)\.pt", os.path.basename(args.ckpt))

    if match:
        epoch = int(match.group(1))

    run_entry = {
        "ckpt": os.path.basename(args.ckpt),
        "epoch": epoch,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "meta": {
            "norm": args.norm,
            "bn_recal": args.bn_recal,
            "batch_size": args.batch_size,
            "loader_workers": args.loader_workers,
            "seed": args.seed,
            "data_dir": args.data_dir,
            "split": "test",
        },
        "overall": result["overall"],
        "per_model_id": result.get("per_model_id"),
        "num_samples": len(labels),
        "summary": {
            "overall": summary_overall,
            "per_model_id": per_model_summary,
        },
    }

    payload = {"runs": []}

    if os.path.exists(save_json):
        try:
            with open(save_json, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)

            payload = (
                loaded
                if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list)
                else {"runs": [loaded]}
            )
        except Exception:
            payload = {"runs": []}

    payload["runs"].append(run_entry)

    with open(save_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\n>>> Saved evaluation results to {save_json}")


if __name__ == "__main__":
    main()