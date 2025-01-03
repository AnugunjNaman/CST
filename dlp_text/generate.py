from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from .config import DLPConfig
from .models import ExpertModels, decode_ids, load_experts, tokenize_batch
from .sampler import DiscreteLangevinSampler


def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _expand_chains(batch: dict[str, torch.Tensor], n_chains: int) -> dict[str, torch.Tensor]:
    if n_chains <= 1:
        return batch
    return {k: v.repeat_interleave(n_chains, dim=0) for k, v in batch.items()}


def _reduce_chains(tokens: torch.Tensor, energy: torch.Tensor, n_src: int, n_chains: int) -> torch.Tensor:
    if n_chains <= 1:
        return tokens
    e = energy.view(n_src, n_chains)
    t = tokens.view(n_src, n_chains, -1)
    idx = e.argmax(dim=1)
    return t[torch.arange(n_src, device=t.device), idx]


def generate_sentences(
    texts: list[str],
    cfg: DLPConfig,
    experts: ExpertModels | None = None,
    sampler: DiscreteLangevinSampler | None = None,
    targets: torch.Tensor | None = None,
    show_progress: bool = True,
) -> list[str]:
    if experts is None:
        experts = load_experts(cfg)
    if sampler is None:
        sampler = DiscreteLangevinSampler(cfg, experts)

    out: list[str] = []
    rng = range(0, len(texts), cfg.batch_size)
    if show_progress:
        rng = tqdm(rng, desc="sample", total=(len(texts) + cfg.batch_size - 1) // cfg.batch_size)

    for start in rng:
        chunk = texts[start : start + cfg.batch_size]
        batch = tokenize_batch(chunk, experts.tokenizer, cfg.max_length, experts.device)
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        if targets is None:
            tgt = sampler.infer_targets(ids, mask)
        else:
            tgt = targets[start : start + cfg.batch_size].to(experts.device)

        n_src = ids.size(0)
        packed = {"input_ids": ids, "attention_mask": mask, "target": tgt, "source": ids}
        packed = _expand_chains(packed, cfg.n_chains)
        result = sampler.run(
            packed["input_ids"],
            packed["attention_mask"],
            packed["source"],
            packed["target"],
        )
        score = result["best_energy"] if cfg.select == "best" else result["energy"]
        chosen = _reduce_chains(result["tokens"], score, n_src, cfg.n_chains)
        out.extend(decode_ids(chosen, experts.tokenizer))
    return out



def generate_file(cfg: DLPConfig) -> list[str]:
    texts = _read_lines(cfg.source_file)
    # Length-bucket: similar-length sentences share a batch (less padding),
    # then restore the original order so src/ref/hyp lines stay paired.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i].split()), reverse=True)
    sorted_hyps = generate_sentences([texts[i] for i in order], cfg)
    hyps = [None] * len(texts)
    for pos, i in enumerate(order):
        hyps[i] = sorted_hyps[pos]
    out_path = Path(cfg.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in hyps:
            f.write(line.strip() + "\n")
    return hyps
