from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import DLPConfig
from .models import ExpertModels, autocast
from .vocab import content_mask


def one_hot(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(tokens, num_classes=vocab_size).to(dtype=torch.float32)


def _content_positions(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """keep (B, L) bool -> (idx (B, P), valid (B, P)). P = max content len."""
    bsz, seqlen = keep.shape
    p = max(int(keep.sum(dim=-1).max().item()), 1)
    idx = keep.long().argsort(dim=1, descending=True)[:, :p]
    valid = keep.gather(1, idx)
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    return idx, valid


def _masked_position_logits(
    experts: ExpertModels,
    embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """BERT PLL: one [MASK] per position, vectorized over the sequence.

    embeds: (B, L, H) -> logits at the masked index: (B, L, V).
    With `keep`, only content positions are forwarded (the expensive part);
    other slots are filled with -1e9 so they can never be proposed.
    """
    bsz, seqlen, hidden = embeds.shape
    if seqlen == 0:
        return embeds.new_zeros(bsz, 0, experts.vocab_size)
    if keep is None:
        idx = torch.arange(seqlen, device=embeds.device).unsqueeze(0).expand(bsz, seqlen)
        valid = torch.ones(bsz, seqlen, dtype=torch.bool, device=embeds.device)
    else:
        idx, valid = _content_positions(keep)
    p = idx.size(1)

    x = embeds.unsqueeze(1).expand(bsz, p, seqlen, hidden).reshape(bsz * p, seqlen, hidden)
    pos = idx.reshape(-1)
    row = torch.arange(bsz * p, device=embeds.device)
    x = x.clone()
    x[row, pos] = experts.mask_embed
    attn = attention_mask.unsqueeze(1).expand(bsz, p, seqlen).reshape(bsz * p, seqlen)
    with autocast(experts):
        logits = experts.mlm(inputs_embeds=x, attention_mask=attn).logits
    picked = logits[row, pos].view(bsz, p, -1).float()

    full = embeds.new_full((bsz, seqlen, experts.vocab_size), -1e9)
    bv, pv = valid.nonzero(as_tuple=True)
    if bv.numel():
        full[bv, idx[bv, pv]] = picked[valid]
    return full


def disc_target_values(
    experts: ExpertModels,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    target: torch.Tensor,
    score: str,
) -> torch.Tensor:
    """U_disc from input ids (no one-hot). Returns (B,)."""
    with autocast(experts):
        logits = experts.disc(input_ids=tokens, attention_mask=attention_mask).logits
    logits = logits.float()
    if score == "logprob":
        values = F.log_softmax(logits, dim=-1)
    elif score == "logits":
        values = logits
    else:
        raise ValueError(f"unknown disc_score={score}")
    return values.gather(1, target.view(-1, 1)).squeeze(1)


def disc_log_density(
    experts: ExpertModels,
    oh: torch.Tensor,
    attention_mask: torch.Tensor,
    target: torch.Tensor,
    score: str,
) -> torch.Tensor:
    """U_disc(s) for the target class. Returns (B,)."""
    embeds = oh @ experts.disc_embed_weight
    with autocast(experts):
        logits = experts.disc(inputs_embeds=embeds, attention_mask=attention_mask).logits
    logits = logits.float()
    if score == "logprob":
        values = F.log_softmax(logits, dim=-1)
    elif score == "logits":
        values = logits
    else:
        raise ValueError(f"unknown disc_score={score}")
    return values.gather(1, target.view(-1, 1)).squeeze(1)


def hamming_penalty(
    oh: torch.Tensor,
    source_ids: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    """sum_i 1[s_i != s0_i] on content tokens. Differentiable in the one-hot."""
    match = oh.gather(-1, source_ids.unsqueeze(-1)).squeeze(-1)
    return ((1.0 - match) * keep).sum(dim=-1)


def hamm_count_energy(h: torch.Tensor, shape: str) -> torch.Tensor:
    """Map edit count -> energy units (before α_hamm)."""
    if shape == "convex":
        return h * (h + 1.0) / 2.0
    if shape == "linear":
        return h
    raise ValueError(f"unknown hamm_shape={shape}")


def sitewise_hamm_delta(
    h_total: torch.Tensor,
    cur_edited: torch.Tensor,
    new_edited: torch.Tensor,
    weight: float,
    shape: str,
    budget: int,
) -> torch.Tensor:
    """Exact Φ(h_new) − Φ(h_cur) for a one-site move, plus a budget wall.

    h_total: current edit count per row. Moving into the wall is forbidden
    (−inf), so the chain is MCMC on the budget-restricted support.
    """
    dn = new_edited.to(h_total.dtype) - cur_edited.to(h_total.dtype)
    if h_total.dim() < dn.dim():
        h_total = h_total.unsqueeze(-1)
    h_total = h_total.expand_as(dn)
    h_new = h_total + dn
    e = weight * dn
    if shape == "convex":
        e = torch.where(
            dn > 0,
            weight * (h_total + 1.0),
            torch.where(dn < 0, -weight * h_total, torch.zeros_like(e)),
        )
    if budget and budget > 0:
        e = torch.where(
            (h_new > float(budget)) & (h_total <= float(budget)),
            e.new_full(e.shape, -1e9),
            e,
        )
    return e


class ProductOfExperts:
    """U(s) = α_mlm U_mlm + α_disc U_disc - α_hamm Hamming(s, s0).

    U is an unnormalized log-density (higher = better), matching Zhang MH.
    """

    def __init__(self, cfg: DLPConfig, experts: ExpertModels):
        self.cfg = cfg
        self.experts = experts

    @torch.no_grad()
    def parts(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        source_ids: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """PLL logits, energy terms, and full U. No BERT backward."""
        oh = one_hot(tokens, self.experts.vocab_size)
        keep_bool = content_mask(source_ids, self.experts.tokenizer)
        keep = keep_bool.to(oh.dtype)
        embeds = oh @ self.experts.embed_weight
        pos_logits = _masked_position_logits(
            self.experts, embeds, attention_mask, keep=keep_bool
        )
        if self.cfg.mlm_score == "logprob":
            mlm_vals = F.log_softmax(pos_logits, dim=-1)
        else:
            mlm_vals = pos_logits
        u_mlm = (oh * mlm_vals * keep.unsqueeze(-1)).sum(dim=(1, 2))
        u_disc = disc_log_density(
            self.experts, oh, attention_mask, target, self.cfg.disc_score
        )
        hamm = hamming_penalty(oh, source_ids, keep)
        u = self.cfg.mlm_weight * u_mlm + self.cfg.disc_weight * u_disc
        if self.cfg.hamm_weight:
            u = u - self.cfg.hamm_weight * hamm_count_energy(hamm, self.cfg.hamm_shape)
        return {
            "u": u,
            "u_mlm": u_mlm,
            "u_disc": u_disc,
            "hamm": hamm,
            "pos_logits": pos_logits,
            "keep": keep.bool(),
        }

    def add_hamming(self, u_experts: torch.Tensor, tokens: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        if not self.cfg.hamm_weight:
            return u_experts
        keep = content_mask(source_ids, self.experts.tokenizer).to(u_experts.dtype)
        oh = one_hot(tokens, self.experts.vocab_size)
        h = hamming_penalty(oh, source_ids, keep)
        return u_experts - self.cfg.hamm_weight * hamm_count_energy(h, self.cfg.hamm_shape)
