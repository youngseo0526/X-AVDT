"""Quick checker: does the reconstructed Classifier match a checkpoint?

Usage:
    python train/verify_ckpt.py --ckpt sa_triplet_dec_bs8_800_es.pt
    python train/verify_ckpt.py --ckpt sa_triplet_dec_bs8_800_es.pt --proj_dim 512

It reports, per parameter:
  - MISSING    : model expects it, checkpoint does not have it
  - UNEXPECTED : checkpoint has it, model does not define it
  - SHAPE      : present in both but shapes differ (with both shapes printed)

If nothing is printed under those headers, the checkpoint loads with strict=True.
The most likely thing to tune is `--proj_dim` (img_proj / attn_proj output dim).
"""
import argparse
import collections

import torch

from utils.network import Classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--norm", default="batch", choices=["batch", "instance", "layer"])
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--attn_in_channels", type=int, default=320)
    p.add_argument("--self_attn", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    model = Classifier(
        self_attn=args.self_attn,
        norm_layer=args.norm,
        attn_in_channels=args.attn_in_channels,
        proj_dim=args.proj_dim,
    )
    model_sd = model.state_dict()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state = collections.OrderedDict(
        (k[7:] if k.startswith("module.") else k, v) for k, v in state.items()
    )

    model_keys = set(model_sd.keys())
    ckpt_keys = set(state.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatch = [
        (k, tuple(model_sd[k].shape), tuple(state[k].shape))
        for k in sorted(model_keys & ckpt_keys)
        if tuple(model_sd[k].shape) != tuple(state[k].shape)
    ]

    print(f"model params: {len(model_keys)} | ckpt params: {len(ckpt_keys)}")
    print(f"MISSING ({len(missing)}):")
    for k in missing:
        print("   ", k, tuple(model_sd[k].shape))
    print(f"UNEXPECTED ({len(unexpected)}):")
    for k in unexpected:
        print("   ", k, tuple(state[k].shape))
    print(f"SHAPE MISMATCH ({len(shape_mismatch)}):")
    for k, ms, cs in shape_mismatch:
        print("   ", k, "model", ms, "vs ckpt", cs)

    if not missing and not unexpected and not shape_mismatch:
        model.load_state_dict(state, strict=True)
        print("\nOK: checkpoint loads with strict=True.")
    else:
        print("\nNOT an exact match yet. See the lists above.")


if __name__ == "__main__":
    main()
