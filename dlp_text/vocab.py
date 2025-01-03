from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def allowed_token_mask(
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    *,
    block_unused: bool = True,
    block_special: bool = True,
    english_only: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean mask over the vocabulary: True = allowed as a proposed token."""
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)
    special_ids = set()
    for attr in (
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
        "unk_token_id",
        "bos_token_id",
        "eos_token_id",
    ):
        tid = getattr(tokenizer, attr, None)
        if tid is not None and tid >= 0:
            special_ids.add(int(tid))
    extra = tokenizer.all_special_ids or []
    special_ids.update(int(i) for i in extra)

    inv = {i: t for t, i in tokenizer.get_vocab().items()}
    for i in range(vocab_size):
        tok = inv.get(i, "")
        if block_special and i in special_ids:
            allowed[i] = False
            continue
        if block_unused and tok.startswith("[unused"):
            allowed[i] = False
            continue
        if english_only and tok:
            core = tok[2:] if tok.startswith("##") else tok
            if core and not all(ord(c) < 128 for c in core):
                allowed[i] = False
    return allowed


def content_mask(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """True on tokens that DLP is allowed to overwrite (not special/pad)."""
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    # [MASK] is writable (prompt filling). Never overwrite pad/cls/sep.
    for attr in ("pad_token_id", "cls_token_id", "sep_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            mask &= input_ids != tid
    return mask
