from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DLPConfig:
    """Hyperparameters. Defaults match Naman (2024) §4 unless noted."""

    task: str = "style_yelp"

    base_model: str = "bert-base-uncased"
    disc_model: str = "textattack/bert-base-uncased-yelp-polarity"
    external_clf: str = "textattack/bert-base-uncased-yelp-polarity"
    grammar_clf: str = "textattack/roberta-base-CoLA"

    # Product-of-experts weights on the log-density U (higher = better).
    # Paper writes energies E then uses MH as if U is log-density; we follow
    # Zhang et al. 2022 and the working sign in the original code: higher U
    # is preferred. α_mlm=1, α_disc=200, α_hamm=60 from §4.
    mlm_weight: float = 1.0
    disc_weight: float = 200.0
    hamm_weight: float = 60.0
    # linear: E = w*h. convex: E = w*h(h+1)/2 — increasing marginal edit cost,
    # so the equilibrium commits the single best edit instead of drifting.
    hamm_shape: str = "linear"
    hamm_budget: int = 0  # hard wall: never exceed this many edits (0 = off)
    mlm_score: str = "logits"  # logits | logprob
    disc_score: str = "logits"  # logits | logprob

    n_flips: int = 4
    num_steps: int = 150
    cycle_length: int = 150
    eta_max: float = 0.07
    eta_min: float = 0.03
    temperature: float = 2.0  # implements the 1/2 in Zhang eq. (2)
    schedule: str = "cosine_minmax"  # cosine_minmax | shifted_cosine | constant
    force_flip: bool = False  # adaptive η + disc top-k leave the MLM mode without banning stay
    top_k: int = 10  # proposal candidates = {current} ∪ MLM top-k, re-scored by true disc
    # sitewise MH: propose up to n_flips one-token substitutions, commit ≤ max_accepts.
    adaptive_eta: bool = True
    target_flip: float = 0.35  # calibrate 1/η so stay vs best-alt hits this (binary softmax)
    target_accept: float = 0.35  # dual-average η scale toward this sitewise accept rate
    eta_adapt_min: float = 0.02
    eta_adapt_max: float = 2.0
    adapt_lr: float = 0.05
    escape_mode: bool = True  # force a flip only where the current token is the MLM argmax
    max_accepts: int = 2  # consider n_flips sites, commit at most this many per step

    max_length: int = 32
    n_chains: int = 1
    select: str = "last"  # last | best
    batch_size: int = 4
    seed: int = 0
    device: str = "cuda:0"

    source_file: str = "data/style_yelp/data.txt"
    ref_file: str = "data/style_yelp/ref.txt"
    output_file: str = "outputs/style_yelp/gen.txt"

    target_label: Optional[int] = None  # if set, force this class; else flip

    block_unused: bool = True
    block_special: bool = True
    english_only: bool = False
    amp: bool = False  # bf16 autocast on the frozen expert forwards

    pos_label: int = 1
    neg_label: int = 0

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "DLPConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        extra = {k: v for k, v in raw.items() if k not in cls.__dataclass_fields__}
        known.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        return d
