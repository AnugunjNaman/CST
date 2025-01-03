from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DLPConfig
from .models import tokenize_batch


def _lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def token_hamming(hyps: list[str], srcs: list[str], tokenizer, max_length: int) -> float:
    dists = []
    for h, s in zip(hyps, srcs):
        ht = tokenizer.encode(h, add_special_tokens=True, truncation=True, max_length=max_length)
        st = tokenizer.encode(s, add_special_tokens=True, truncation=True, max_length=max_length)
        n = max(len(ht), len(st))
        ht = ht + [tokenizer.pad_token_id] * (n - len(ht))
        st = st + [tokenizer.pad_token_id] * (n - len(st))
        dists.append(sum(a != b for a, b in zip(ht, st)))
    return float(sum(dists) / max(len(dists), 1))


def distinct_n(hyps: list[str], n: int) -> float:
    total = 0
    uniq: set[tuple[str, ...]] = set()
    for sent in hyps:
        toks = sent.lower().split()
        grams = [tuple(toks[i : i + n]) for i in range(0, max(len(toks) - n + 1, 0))]
        total += len(grams)
        uniq.update(grams)
    return float(len(uniq) / total) if total else 0.0


def bertscore_f1(hyps: list[str], srcs: list[str], device: str) -> float:
    try:
        import bert_score

        _, _, f1 = bert_score.score(hyps, srcs, lang="en", device=device, verbose=False)
        return float(f1.mean().item())
    except Exception:
        return _fallback_bertscore(hyps, srcs, device)


def _fallback_bertscore(hyps: list[str], srcs: list[str], device: str) -> float:
    """Mean-pooled BERT cosine, used only if `bert_score` is not installed."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    scores = []
    with torch.no_grad():
        for h, s in zip(hyps, srcs):
            batch = tok([h, s], padding=True, truncation=True, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            a, b = pooled[0], pooled[1]
            scores.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    return float(sum(scores) / max(len(scores), 1))


def classifier_accuracy(
    texts: list[str],
    targets: list[int],
    model_name: str,
    device: str,
    batch_size: int = 16,
    max_length: int = 32,
) -> float:
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tgt = targets[i : i + batch_size]
            enc = tokenize_batch(chunk, tok, max_length, torch.device(device))
            pred = clf(**enc).logits.argmax(-1).tolist()
            correct += sum(int(p == t) for p, t in zip(pred, tgt))
    return 100.0 * correct / max(len(texts), 1)


def grammar_score(texts: list[str], model_name: str, device: str, batch_size: int = 16) -> float:
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    good = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            pred = clf(**enc).logits.argmax(-1)
            # CoLA: label 1 = grammatical
            good += int((pred == 1).sum().item())
    return good / max(len(texts), 1)


def infer_style_targets(
    srcs: list[str],
    model_name: str,
    device: str,
    max_length: int,
) -> list[int]:
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    targets = []
    with torch.no_grad():
        for i in range(0, len(srcs), 16):
            chunk = srcs[i : i + 16]
            enc = tokenize_batch(chunk, tok, max_length, torch.device(device))
            src_lab = clf(**enc).logits.argmax(-1)
            targets.extend(((src_lab + 1) % 2).tolist())
    return targets


def evaluate(cfg: DLPConfig, hyp_file: str | None = None) -> dict[str, float]:
    hyps = _lines(hyp_file or cfg.output_file)
    srcs = _lines(cfg.source_file)
    refs_path = Path(cfg.ref_file)
    refs = _lines(cfg.ref_file) if refs_path.exists() else None
    n = min(len(hyps), len(srcs))
    hyps, srcs = hyps[:n], srcs[:n]
    if refs is not None:
        refs = refs[:n]

    device = cfg.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    targets = infer_style_targets(srcs, cfg.disc_model, device, cfg.max_length)

    metrics: dict[str, float] = {
        "n": float(n),
        "hamming_src": token_hamming(hyps, srcs, tok, cfg.max_length),
        "bertscore_src": bertscore_f1(hyps, srcs, device),
        "ext_clf": classifier_accuracy(hyps, targets, cfg.external_clf, device, max_length=cfg.max_length),
        "int_clf": classifier_accuracy(hyps, targets, cfg.disc_model, device, max_length=cfg.max_length),
        "distinct_1": distinct_n(hyps, 1),
        "distinct_2": distinct_n(hyps, 2),
        "distinct_3": distinct_n(hyps, 3),
        "grammar": grammar_score(hyps, cfg.grammar_clf, device),
    }
    if refs is not None:
        metrics["bleu_ref"] = corpus_bleu(hyps, refs)
        metrics["hamming_ref"] = token_hamming(refs, srcs, tok, cfg.max_length)
    return metrics


def format_metrics(m: dict[str, float]) -> str:
    order = [
        ("n", "N"),
        ("bleu_ref", "BLEU (ref)"),
        ("bertscore_src", "BERTScore (src)"),
        ("hamming_src", "Hamming (src)"),
        ("hamming_ref", "Hamming (ref vs src)"),
        ("int_clf", "Int. Clsf."),
        ("ext_clf", "Ext. Clsf."),
        ("grammar", "Grammaticality"),
        ("distinct_1", "Distinct-1"),
        ("distinct_2", "Distinct-2"),
        ("distinct_3", "Distinct-3"),
    ]
    lines = []
    for k, name in order:
        if k not in m:
            continue
        val = m[k]
        if k in {"bertscore_src", "grammar", "distinct_1", "distinct_2", "distinct_3"}:
            lines.append(f"{name:24s} {val:.4f}")
        elif k == "n":
            lines.append(f"{name:24s} {int(val)}")
        else:
            lines.append(f"{name:24s} {val:.2f}")
    return "\n".join(lines)
