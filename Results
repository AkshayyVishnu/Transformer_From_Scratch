# Results

All numbers from `ablations.py` and `gpt.py`. Raw data in
`ablation_results.csv` and `fullscale_curve.csv`.

---

## Full-scale model — 10,715,201 parameters

**Configuration:** n_embd 384, n_head 6 (head_size 64), n_layer 6,
block_size 256, batch_size 64, max_iters 5000, AdamW lr 3e-4, dropout 0.2.

| Step | Train | Val |
|---|---|---|
| 0 | 4.2825 | 4.2921 |
| 500 | 1.7180 | 1.8609 |
| 1000 | 1.5210 | 1.7012 |
| 1500 | 1.4234 | 1.6228 |
| 2000 | 1.3664 | 1.5808 |
| 2500 | 1.3283 | 1.5607 |
| 3000 | 1.2941 | 1.5404 |
| 3500 | 1.2671 | 1.5197 |
| 4000 | 1.2417 | 1.5201 |
| **4500** | 1.2210 | **1.5155** ← best |
| 4999 | 1.2020 | 1.5217 |

- Step-0 val loss 4.2921 vs uniform baseline `−ln(1/65) = 4.174` — correct
  initialisation.
- Best val 1.5155 at step 4500; val rises afterward while train continues to
  fall. Overfitting onset.
- Train/val gap: 0.14 at step 500 → 0.32 at step 4999.

---

## Component ablations — 0.8M scale

**Configuration:** n_embd 128, n_head 4 (head_size 32), n_layer 4,
block_size 64, batch_size 32, max_iters 3000, AdamW lr 1e-3, **dropout 0.0**,
seed 1337 for every run.

| Configuration | Val loss | Δ vs baseline | Params | Param-neutral |
|---|---|---|---|---|
| baseline | 1.5923 | — | 816,705 | — |
| no positional embeddings | 1.6454 | +0.0531 | 816,705 | yes |
| no FFN | 1.7194 | +0.1271 | 289,857 | **no** (−526,848) |
| **no residual connections** | **3.3516** | **+1.7593** | 816,705 | yes |
| no LayerNorm | 1.6312 | +0.0389 | 814,401 | **no** (−2,304) |
| no 1/√d scaling | 1.6077 | +0.0154 | 816,705 | yes |
| single head (width 128) | 1.6181 | +0.0258 | 816,705 | yes |
| **no causal mask** | **0.0327** | **−1.5596** | 816,705 | yes |

Six of eight are parameter-neutral. The `no FFN` delta conflates the loss of
per-token nonlinearity with a 65% capacity reduction and should be read as an
upper bound.

---

## Predictions vs. outcomes

Recorded before any ablation was run.

| Configuration | Predicted | Actual | Error | Verdict |
|---|---|---|---|---|
| baseline | 1.590 | 1.5923 | +0.002 | ✓ |
| no_pos_emb | 1.630 | 1.6454 | +0.015 | ✓ close |
| no_ffn | 1.610 | 1.7194 | +0.109 | ✗ underestimated |
| no_residual | 1.654 | 3.3516 | +1.698 | ✗✗ badly underestimated |
| no_layernorm | 1.640 | 1.6312 | −0.009 | ✓ |
| no_scaling | 1.900 | 1.6077 | −0.292 | ✗✗ badly overestimated |
| single_head | 1.640 | 1.6181 | −0.022 | ✓ |
| no_causal_mask | 0.600 | 0.0327 | −0.567 | ✓ direction, ✗ magnitude |

5 of 8 within 0.03. The two large misses (`no_residual`, `no_scaling`) are
analysed in the main README.

---

## Reference points

| Model | Context | Val loss |
|---|---|---|
| Uniform over 65 characters | — | 4.174 |
| Bigram model | 1 token | ~2.50 |
| No-residual transformer (0.8M) | 64 tokens | 3.3516 |
| Baseline transformer (0.8M) | 64 tokens | 1.5923 |
| **Full-scale transformer (10.7M)** | 256 tokens | **1.5155** |

Note that the no-residual transformer performs worse than a bigram model
despite having 816,705 parameters and 64 tokens of context.

---

## Scale versus architecture

| Change | Δ val loss |
|---|---|
| 0.8M → 10.7M params, 64 → 256 context | **−0.077** |
| Removing residual connections (at 0.8M) | **+1.759** |
| Removing the FFN | +0.127 |
| Removing positional embeddings | +0.053 |

A 13× parameter increase plus 4× context bought 0.077. Removing residuals cost
23× that.

**Supporting evidence that the task is data-bound:** widening the 0.8M baseline
from n_embd 128 to 132 (+75,904 parameters) produced *worse* validation loss at
the same iteration count — 1.6251 vs 1.6157.

---

## Measured seed variance

From a separate multi-seed run at the ablation configuration (5 seeds:
1337, 42, 7, 2024, 91):

| Depth | Val loss (mean ± std) |
|---|---|
| 4 layers | 1.6432 ± 0.0105 |
| 8 layers | 1.6062 ± 0.0039 |

**Noise floor: ±0.004 to ±0.011.** Effects above roughly 0.02 are
distinguishable from run-to-run variance. Of the ablations above, all except
`no_scaling` (+0.0154) clear that threshold. That result should be treated as
suggestive rather than established.

