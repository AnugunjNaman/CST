# DLP Text Style Transfer

Reference implementation of the discrete Langevin style-transfer sampler from
[*Customized Style Transfer using Discrete Sampling*](https://aclanthology.org/2024.customnlp4u-1.12/)
(Anugunj Naman, CustomNLP4U 2024):
a product-of-experts energy `U = α_mlm·BERT-MLM + α_disc·sentiment − Φ(Hamming)`
sampled with Metropolis–Hastings, with per-site adaptive step size and convex
Hamming pricing. Self-contained; PyTorch only.

On the full 1000-sentence Yelp test it reproduces the paper's Table 1 profile:
BLEU 22.93 (21.19) · BERTScore 0.958 (0.97) · Hamming 1.43 (1.23) · Clsf. 97.4 (85.2),
at ~0.37 s/sample on 2× L40S.

If you use this code, please cite the paper above.

## Implementation notes

A few places where the printed equations need interpretation; this repo implements
the interpretations that work well in practice:

- The cyclical step-size schedule (eq. 10) is the standard cosine cycle between
  η_min and η_max; the printed expression omits a parenthesization that would take
  η outside that range.
- Temperature 2.0 absorbs the 1/2 factor in the one-hot Langevin gradient (eq. 8).
- The per-site proposal uses masked-LM logits at the active position; for text we
  found one-hot autograd through the full-sequence energy to be an imprecise
  coordinate factor.
- The discriminator is evaluated on the actual one-token substitutions in the
  proposal (MLM top-k = 30) rather than through its embedding gradient.
- The Hamming term enters the MH acceptance test only, with a convex schedule
  Φ(h) = w·h(h+1)/2, which keeps 1–2 edits per sentence.
- Sequences are padded to the batch maximum (max_length 32).

## Run

```bash
pip install -r requirements.txt
python main.py --config configs/style_yelp.yaml                 # generate (1000 sentences)
python main.py --config configs/style_yelp.yaml --eval_only     # evaluate outputs/style_yelp/gen.txt
```

Optional 2-GPU data parallel:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/style_yelp.yaml --shard 0 --num_shards 2 &
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/style_yelp.yaml --shard 1 --num_shards 2 &
wait
paste -d'\n' outputs/style_yelp/gen.txt.shard0of2 outputs/style_yelp/gen.txt.shard1of2 > outputs/style_yelp/gen.txt
```

## Citation

```bibtex
@inproceedings{naman-2024-customized,
    title = "Customized Style Transfer using Discrete Sampling",
    author = "Naman, Anugunj",
    booktitle = "Proceedings of the 1st Workshop on Customizable {NLP}: Progress and Challenges in Customizing {NLP} for a Domain, Application, Group, or Individual ({C}ustom{NLP}4{U})",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.customnlp4u-1.12/",
    doi = "10.18653/v1/2024.customnlp4u-1.12",
    pages = "150--155"
}
```
