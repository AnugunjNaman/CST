from __future__ import annotations

import math

import torch
from torch.distributions import Categorical

from .config import DLPConfig
from .energy import ProductOfExperts, disc_target_values, sitewise_hamm_delta
from .models import ExpertModels
from .vocab import content_mask


def cyclical_step_size(step: int, cfg: DLPConfig) -> float:
    """Paper §3.2.1. The published formula is missing parentheses.

    Written:  η_k = max(η_max · cos(π mod(k,K)/K) + 1, η_min)
    Intended cyclical SG-MCMC: oscillate between η_min and η_max.
    """
    k = cfg.cycle_length if cfg.cycle_length else cfg.num_steps
    k = max(int(k), 1)
    phase = math.pi * ((step % k) / k)
    cos = math.cos(phase)
    if cfg.schedule == "constant":
        return float(cfg.eta_max)
    if cfg.schedule == "shifted_cosine":
        # η = η_max (cos + 2)  ∈ [η_max, 3 η_max]
        return float(cfg.eta_max * cos + 2.0 * cfg.eta_max)
    # cosine_minmax: η ∈ [η_min, η_max]
    return float(cfg.eta_min + 0.5 * (cfg.eta_max - cfg.eta_min) * (1.0 + cos))


def _inv_eta(eta: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """1/η broadcast to ref's spatial shape (everything except vocab)."""
    spatial = ref.shape[:-1]
    if not torch.is_tensor(eta):
        return ref.new_full(spatial, 1.0 / max(float(eta), 1e-8))
    out = eta.to(device=ref.device, dtype=ref.dtype)
    while out.ndim < len(spatial):
        out = out.unsqueeze(-1)
    if out.shape != spatial:
        out = out.expand(spatial)
    return 1.0 / out.clamp(min=1e-8)


def dlp_logits(
    grad: torch.Tensor,
    tokens: torch.Tensor,
    eta: float | torch.Tensor,
    temperature: float,
    allowed: torch.Tensor | None = None,
    force_flip: bool = False,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Categorical DLP logits (Zhang 2022 eq. 2 / Naman eq. 8).

    For one-hot coordinates:
      (1/2) ∇U_i · (e_c - e_{s_i}) - ||e_c - e_{s_i}||² / (2η)
    = (∇U_{i,c} - ∇U_{i,s_i}) / 2  -  1{c ≠ s_i} / η

    Temperature 2 implements the 1/2, as in the categorical DLP derivation.
    `eta` may be a scalar or a tensor matching the spatial dims of `grad`.
    """
    g = grad / max(float(temperature), 1e-8)
    current = g.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    first = g - current.unsqueeze(-1)
    penalty = _inv_eta(eta, g).unsqueeze(-1).expand_as(g).clone()
    penalty.scatter_(-1, tokens.unsqueeze(-1), 0.0)
    logits = first - penalty
    if allowed is not None:
        logits = logits.masked_fill(~allowed.view(1, 1, -1), -1e9)
        if not force_flip:
            logits.scatter_(-1, tokens.unsqueeze(-1), first.gather(-1, tokens.unsqueeze(-1)))
    if force_flip:
        # 1/η is constant on all c ≠ s_i, so softmax among alternatives is just ∇U.
        ban = torch.zeros_like(logits, dtype=torch.bool)
        ban.scatter_(-1, tokens.unsqueeze(-1), True)
        if active is not None:
            ban = ban & active.unsqueeze(-1)
        logits = logits.masked_fill(ban, -1e9)
    return logits


def adaptive_eta(
    grad: torch.Tensor,
    tokens: torch.Tensor,
    temperature: float,
    target_flip: float,
    eta_min: float,
    eta_max: float,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-site η so stay vs best-alternative has flip probability ≈ target_flip.

    Binary softmax: p_flip = sigmoid( (g_alt - g_cur)/τ - 1/η ).
    Paper η ∈ [0.03, 0.07] is the wrong scale for BERT logits (~10–20); this
    puts 1/η on the same scale as the local gap so the Langevin stay term
    actually interpolates instead of freezing or being a no-op.
    """
    tau = max(float(temperature), 1e-8)
    g = grad / tau
    g_cur = g.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    banned = torch.zeros_like(g, dtype=torch.bool)
    banned.scatter_(-1, tokens.unsqueeze(-1), True)
    g_alt = g.masked_fill(banned, -1e9).max(dim=-1).values
    gap = g_alt - g_cur
    pi = min(max(float(target_flip), 1e-4), 1.0 - 1e-4)
    logit_pi = math.log(pi) - math.log(1.0 - pi)
    inv = (gap - logit_pi).clamp(min=1e-6)
    eta = (1.0 / inv).clamp(min=float(eta_min), max=float(eta_max))
    if active is not None:
        eta = torch.where(active, eta, eta.new_full(eta.shape, float(eta_min)))
    return eta


def sample_active_idx(
    keep: torch.Tensor,
    n_flips: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (active mask, indices). Extra slots are invalid if a row has < n content tokens."""
    bsz, seqlen = keep.shape
    n = max(int(n_flips), 1)
    n = min(n, seqlen)
    scores = torch.rand(bsz, seqlen, device=keep.device)
    scores = scores.masked_fill(~keep, -1e9)
    _, idx = scores.topk(n, dim=-1)
    n_keep = keep.sum(dim=-1)
    valid = torch.arange(n, device=keep.device).unsqueeze(0) < n_keep.unsqueeze(1)
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    active = torch.zeros_like(keep)
    b_ix, f_ix = valid.nonzero(as_tuple=True)
    if b_ix.numel():
        active[b_ix, idx[b_ix, f_ix]] = True
    return active & keep, idx


def pack_site_trials(
    tokens: torch.Tensor,
    idx: torch.Tensor,
    new_ids: torch.Tensor,
) -> torch.Tensor:
    """(B, L) + (B, F) sites → (B*F, L) one-site substitutions."""
    bsz, seqlen = tokens.shape
    n_f = idx.size(1)
    trials = tokens.unsqueeze(1).repeat(1, n_f, 1)
    b = torch.arange(bsz, device=tokens.device).unsqueeze(1)
    f = torch.arange(n_f, device=tokens.device).unsqueeze(0)
    trials[b, f, idx] = new_ids
    return trials.reshape(bsz * n_f, seqlen)


def cap_accepts(
    delta_u: torch.Tensor,
    passed: torch.Tensor,
    max_accepts: int,
) -> torch.Tensor:
    """Keep at most `max_accepts` MH-passing sites, highest ΔU first."""
    if max_accepts is None or max_accepts <= 0 or max_accepts >= passed.size(-1):
        return passed
    scores = delta_u.masked_fill(~passed, -1e9)
    k = min(int(max_accepts), passed.size(-1))
    _, top = scores.topk(k, dim=-1)
    kept = torch.zeros_like(passed)
    top_ok = scores.gather(1, top) > -1e8
    b = torch.arange(passed.size(0), device=passed.device).unsqueeze(1).expand_as(top)
    kept[b[top_ok], top[top_ok]] = True
    return kept


def merge_sitewise(
    tokens: torch.Tensor,
    idx: torch.Tensor,
    new_ids: torch.Tensor,
    accept: torch.Tensor,
) -> torch.Tensor:
    """Apply independently accepted substitutions. idx must not collide."""
    out = tokens.clone()
    b_ix, f_ix = accept.nonzero(as_tuple=True)
    if b_ix.numel():
        out[b_ix, idx[b_ix, f_ix]] = new_ids[b_ix, f_ix]
    return out


class DiscreteLangevinSampler:
    def __init__(self, cfg: DLPConfig, experts: ExpertModels):
        self.cfg = cfg
        self.experts = experts
        self.energy = ProductOfExperts(cfg, experts)
        self._eta_scale = 1.0

    @torch.no_grad()
    def infer_targets(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.experts.disc(input_ids=input_ids, attention_mask=attention_mask).logits
        src = logits.argmax(dim=-1)
        if self.cfg.target_label is not None:
            return torch.full_like(src, int(self.cfg.target_label))
        nclass = logits.size(-1)
        return (src + 1) % nclass if nclass == 2 else src

    def _proposal_grad(self, tokens, attention_mask, source_ids, target, parts=None):
        if parts is None:
            parts = self.energy.parts(tokens, attention_mask, source_ids, target)
        return parts["pos_logits"], parts

    def _mlm_values(self, pos_logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.mlm_score == "logprob":
            return torch.nn.functional.log_softmax(pos_logits, dim=-1)
        return pos_logits

    def _eta_for(self, grad: torch.Tensor, tokens: torch.Tensor, step_idx: int, active: torch.Tensor) -> torch.Tensor:
        cycle = cyclical_step_size(step_idx, self.cfg)
        if not self.cfg.adaptive_eta:
            return grad.new_full(tokens.shape, cycle)
        eta = adaptive_eta(
            grad,
            tokens,
            temperature=self.cfg.temperature,
            target_flip=self.cfg.target_flip,
            eta_min=self.cfg.eta_adapt_min,
            eta_max=self.cfg.eta_adapt_max,
            active=active,
        )
        # Paper cosine still anneals: high at cycle ends, low at the midpoint.
        eta = eta * (cycle / max(float(self.cfg.eta_max), 1e-8))
        eta = eta * float(self._eta_scale)
        return eta.clamp(min=self.cfg.eta_adapt_min, max=self.cfg.eta_adapt_max)

    def _bump_eta_scale(self, accept: torch.Tensor, proposed: torch.Tensor) -> None:
        if not self.cfg.adaptive_eta or proposed.numel() == 0:
            return
        rate = float(accept.float().mean().clamp(min=0.0, max=1.0))
        lr = float(self.cfg.adapt_lr)
        target = float(self.cfg.target_accept)
        # High accept → larger η (more exploration), MALA-style dual averaging.
        self._eta_scale *= math.exp(lr * (rate - target))
        self._eta_scale = min(max(self._eta_scale, 0.25), 4.0)

    def _propose_disc_topk(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        source_ids: torch.Tensor,
        target: torch.Tensor,
        idx: torch.Tensor,
        valid: torch.Tensor,
        parts: dict[str, torch.Tensor],
        eta_bf: torch.Tensor,
        ban_slot0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DLP over {current} ∪ MLM-top-k, with true disc scores on each candidate.

        Embedding ∇disc is wayward (points at 'approached', not 'better').
        Evaluating the classifier on the actual one-token substitution is cheap
        relative to PLL and is the coordinate factor Zhang's expansion wants.
        """
        bsz, n_f = idx.shape
        k = max(int(self.cfg.top_k), 1)
        b = torch.arange(bsz, device=tokens.device).unsqueeze(1)
        pos_logits = parts["pos_logits"]
        if self.experts.allowed is not None:
            pos_logits = pos_logits.masked_fill(~self.experts.allowed.view(1, 1, -1), -1e9)
        _, top_ids = pos_logits.topk(k, dim=-1)
        cur_at = tokens.gather(1, idx)
        alt_ids = top_ids[b, idx]
        cands = torch.cat([cur_at.unsqueeze(-1), alt_ids], dim=-1)
        n_c = cands.size(-1)

        mlm_vals = self._mlm_values(parts["pos_logits"])
        site_mlm = mlm_vals[b.unsqueeze(-1), idx.unsqueeze(-1), cands]

        src_at = source_ids.gather(1, idx)
        site_hamm = (cands != src_at.unsqueeze(-1)).to(mlm_vals.dtype)

        site_disc = parts["u_disc"].unsqueeze(-1).unsqueeze(-1).expand(bsz, n_f, n_c).clone()
        alt = cands[:, :, 1:]
        alt_trials = self._pack_candidate_trials(tokens, idx, alt)
        attn_rep = attention_mask.repeat_interleave(n_f * (n_c - 1), dim=0)
        tgt_rep = target.repeat_interleave(n_f * (n_c - 1), dim=0)
        disc_alt = disc_target_values(
            self.experts, alt_trials, attn_rep, tgt_rep, self.cfg.disc_score
        ).view(bsz, n_f, n_c - 1)
        site_disc[:, :, 1:] = disc_alt

        scores = (
            self.cfg.mlm_weight * site_mlm
            + self.cfg.disc_weight * site_disc
            - self.cfg.hamm_weight * site_hamm
        )
        # Duplicate of current inside top-k: knock it out so stay mass sits on slot 0.
        dup = (cands[:, :, 1:] == cur_at.unsqueeze(-1)) & valid.unsqueeze(-1)
        scores = scores.clone()
        scores[:, :, 1:] = scores[:, :, 1:].masked_fill(dup, -1e9)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1e9)
        scores[:, :, 0] = scores[:, :, 0].masked_fill(~valid, 0.0)

        cur_idx = torch.zeros(bsz, n_f, dtype=torch.long, device=tokens.device)
        logits = dlp_logits(
            scores,
            cur_idx,
            eta_bf,
            self.cfg.temperature,
            force_flip=False,
            active=valid,
        )
        if ban_slot0 is not None:
            logits = logits.clone()
            logits[:, :, 0] = logits[:, :, 0].masked_fill(ban_slot0 & valid, -1e9)
        dist = Categorical(logits=logits)
        choice = dist.sample()
        choice = torch.where(valid, choice, cur_idx)
        proposed = cands.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        log_q_fwd = dist.log_prob(choice)

        # Reverse DLP on the same frozen candidate table (locally balanced).
        logits_rev = dlp_logits(
            scores,
            choice,
            eta_bf,
            self.cfg.temperature,
            force_flip=False,
            active=valid,
        )
        if ban_slot0 is not None:
            logits_rev = logits_rev.clone()
            logits_rev.scatter_(
                -1,
                choice.unsqueeze(-1),
                torch.where(
                    (ban_slot0 & valid).unsqueeze(-1),
                    logits_rev.new_full(choice.unsqueeze(-1).shape, -1e9),
                    logits_rev.gather(-1, choice.unsqueeze(-1)),
                ),
            )
        log_q_rev = Categorical(logits=logits_rev).log_prob(cur_idx)
        log_q_fwd = torch.where(valid, log_q_fwd, torch.zeros_like(log_q_fwd))
        log_q_rev = torch.where(valid, log_q_rev, torch.zeros_like(log_q_rev))
        chosen_scores = scores.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        stay_scores = scores[:, :, 0]
        delta_u = (chosen_scores - stay_scores).masked_fill(~valid, 0.0)
        return proposed, log_q_fwd, log_q_rev, delta_u

    def _pack_candidate_trials(
        self,
        tokens: torch.Tensor,
        idx: torch.Tensor,
        alt: torch.Tensor,
    ) -> torch.Tensor:
        """tokens (B,L), idx (B,F), alt (B,F,K) → (B*F*K, L)."""
        bsz, seqlen = tokens.shape
        n_f, n_k = alt.shape[1], alt.shape[2]
        trials = tokens.unsqueeze(1).unsqueeze(1).repeat(1, n_f, n_k, 1)
        b = torch.arange(bsz, device=tokens.device)[:, None, None].expand(bsz, n_f, n_k)
        f = torch.arange(n_f, device=tokens.device)[None, :, None].expand(bsz, n_f, n_k)
        kk = torch.arange(n_k, device=tokens.device)[None, None, :].expand(bsz, n_f, n_k)
        pos = idx.unsqueeze(-1).expand(bsz, n_f, n_k)
        trials[b, f, kk, pos] = alt
        return trials.reshape(bsz * n_f * n_k, seqlen)

    def step(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        source_ids: torch.Tensor,
        target: torch.Tensor,
        step_idx: int,
        keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One sitewise MH step: F independent one-site proposals, capped commits."""
        grad, parts = self._proposal_grad(tokens, attention_mask, source_ids, target)
        u = parts["u"]
        active, idx = sample_active_idx(keep, self.cfg.n_flips)
        n_f = idx.size(1)
        bsz = tokens.size(0)
        n_keep = keep.sum(dim=-1)
        valid = torch.arange(n_f, device=tokens.device).unsqueeze(0) < n_keep.unsqueeze(1)
        eta_bl = self._eta_for(grad, tokens, step_idx, active)
        eta_bf = eta_bl.gather(1, idx)
        at_mode = tokens == grad.argmax(dim=-1)
        if self.cfg.force_flip:
            ban_stay = active
        elif self.cfg.escape_mode:
            # Flip only sites already sitting on the MLM argmax (else stay mass is right).
            ban_stay = active & at_mode
        else:
            ban_stay = torch.zeros_like(active)
        use_force = bool(self.cfg.force_flip or self.cfg.escape_mode)

        proposed, log_q_fwd, log_q_rev, delta_u = self._propose_disc_topk(
            tokens,
            attention_mask,
            source_ids,
            target,
            idx,
            valid,
            parts,
            eta_bf,
            ban_slot0=ban_stay.gather(1, idx) if use_force else None,
        )
        log_acc = delta_u + log_q_rev - log_q_fwd
        accept = torch.rand_like(log_acc).log() < log_acc.clamp(max=0.0)
        same = proposed == tokens.gather(1, idx)
        changed = accept & valid & ~same
        accept = cap_accepts(delta_u, changed, self.cfg.max_accepts) | (same & valid)
        self._bump_eta_scale(accept & valid & (proposed != tokens.gather(1, idx)), proposed)

        out = merge_sitewise(tokens, idx, proposed, accept)
        hops = (out != tokens) & keep
        # Rebuild U from accepted deltas (independent, first-order additive).
        u_out = u + (delta_u * accept.float()).sum(dim=-1)
        return out, u_out, hops.float().sum(-1)

    def run(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        source_ids: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        keep = content_mask(source_ids, self.experts.tokenizer)
        cur = tokens.clone()
        best = tokens.clone()
        best_u = torch.full((tokens.size(0),), -1e30, device=tokens.device, dtype=torch.float32)
        acc_hops = []
        last_u = None
        self._eta_scale = 1.0
        for t in range(self.cfg.num_steps):
            cur, u, hops = self.step(cur, attention_mask, source_ids, target, t, keep)
            last_u = u
            acc_hops.append(hops)
            better = u > best_u
            best_u = torch.where(better, u, best_u)
            best = torch.where(better.unsqueeze(-1), cur, best)
        chosen = best if self.cfg.select == "best" else cur
        hops_mean = torch.stack(acc_hops, dim=0).mean(0) if acc_hops else torch.zeros_like(best_u)
        return {
            "tokens": chosen,
            "energy": last_u if last_u is not None else best_u,
            "best_energy": best_u,
            "hops": hops_mean,
        }
