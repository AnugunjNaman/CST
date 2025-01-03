from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from .config import DLPConfig
from .vocab import allowed_token_mask


@dataclass
class ExpertModels:
    tokenizer: PreTrainedTokenizerBase
    mlm: nn.Module
    disc: nn.Module
    embed_weight: torch.Tensor
    disc_embed_weight: torch.Tensor
    mask_embed: torch.Tensor
    mask_id: int
    vocab_size: int
    allowed: torch.Tensor
    device: torch.device
    amp: bool = False


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_experts(cfg: DLPConfig) -> ExpertModels:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    mlm = _freeze(AutoModelForMaskedLM.from_pretrained(cfg.base_model)).to(device)
    disc = _freeze(AutoModelForSequenceClassification.from_pretrained(cfg.disc_model)).to(
        device
    )

    embed_weight = mlm.get_input_embeddings().weight
    disc_embed_weight = disc.get_input_embeddings().weight
    mask_id = int(tokenizer.mask_token_id)
    mask_embed = embed_weight[mask_id]
    vocab_size = int(mlm.config.vocab_size)
    allowed = allowed_token_mask(
        tokenizer,
        vocab_size,
        block_unused=cfg.block_unused,
        block_special=cfg.block_special,
        english_only=cfg.english_only,
        device=device,
    )
    return ExpertModels(
        tokenizer=tokenizer,
        mlm=mlm,
        disc=disc,
        embed_weight=embed_weight,
        disc_embed_weight=disc_embed_weight,
        mask_embed=mask_embed,
        mask_id=mask_id,
        vocab_size=vocab_size,
        allowed=allowed,
        device=device,
        amp=bool(getattr(cfg, "amp", False)),
    )


def autocast(experts: ExpertModels):
    """bf16 autocast for the frozen expert forwards (no-op on CPU / when off)."""
    enabled = bool(getattr(experts, "amp", False)) and experts.device.type == "cuda"
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def tokenize_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def decode_ids(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
