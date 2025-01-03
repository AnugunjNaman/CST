#!/usr/bin/env python3
"""Discrete Langevin style transfer (Naman, CustomNLP4U 2024)."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from dlp_text.config import DLPConfig
from dlp_text.evaluate import evaluate, format_metrics
from dlp_text.generate import generate_file


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/style_yelp.yaml")
    p.add_argument("--source_file", default=None)
    p.add_argument("--ref_file", default=None)
    p.add_argument("--output_file", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--n_flips", type=int, default=None)
    p.add_argument("--n_chains", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N source lines")
    p.add_argument(
        "--shard", type=int, default=None,
        help="process lines [shard::num_shards] (data-parallel across GPUs)",
    )
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval", action="store_true", dest="run_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "source_file": args.source_file,
        "ref_file": args.ref_file,
        "output_file": args.output_file,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "n_flips": args.n_flips,
        "n_chains": args.n_chains,
        "seed": args.seed,
    }
    cfg = DLPConfig.from_yaml(args.config, **overrides)
    _seed(cfg.seed)

    if args.limit is not None:
        src = Path(cfg.source_file).read_text(encoding="utf-8").splitlines()
        tmp = Path("/tmp") / f"dlp_src_{cfg.seed}.txt"
        tmp.write_text("\n".join(src[: args.limit]) + "\n", encoding="utf-8")
        cfg.source_file = str(tmp)
        if Path(cfg.ref_file).exists():
            refs = Path(cfg.ref_file).read_text(encoding="utf-8").splitlines()
            rtmp = Path("/tmp") / f"dlp_ref_{cfg.seed}.txt"
            rtmp.write_text("\n".join(refs[: args.limit]) + "\n", encoding="utf-8")
            cfg.ref_file = str(rtmp)

    if args.shard is not None and args.num_shards:
        s, n = int(args.shard), int(args.num_shards)
        src = Path(cfg.source_file).read_text(encoding="utf-8").splitlines()
        refs = (
            Path(cfg.ref_file).read_text(encoding="utf-8").splitlines()
            if Path(cfg.ref_file).exists()
            else []
        )
        src_s = src[s::n]
        refs_s = refs[s::n] if refs else []
        stem = Path(cfg.output_file)
        cfg.output_file = str(stem.parent / f"{stem.name}.shard{s}of{n}")
        tmp = Path("/tmp") / f"dlp_src_shard{s}of{n}.txt"
        tmp.write_text("\n".join(src_s) + "\n", encoding="utf-8")
        cfg.source_file = str(tmp)
        if refs_s:
            rtmp = Path("/tmp") / f"dlp_ref_shard{s}of{n}.txt"
            rtmp.write_text("\n".join(refs_s) + "\n", encoding="utf-8")
            cfg.ref_file = str(rtmp)
        else:
            cfg.ref_file = "/tmp/dlp_no_refs.txt"

    if not args.eval_only:
        t0 = time.time()
        hyps = generate_file(cfg)
        elapsed = time.time() - t0
        per = elapsed / max(len(hyps), 1)
        print(f"wrote {len(hyps)} sentences to {cfg.output_file}")
        print(f"time {elapsed:.1f}s  ({per:.2f} s/sample)")

    if args.eval_only or args.run_eval:
        metrics = evaluate(cfg, hyp_file=cfg.output_file)
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
