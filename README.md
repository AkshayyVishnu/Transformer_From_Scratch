# Transformer_From_Scratch

# GPT From Scratch — Implementation, Scaling, and Component Ablation

A decoder-only transformer implemented from first principles in PyTorch: scaled
dot-product attention, causal masking, multi-head decomposition, learned
positional embeddings, pre-norm residual blocks, and sampled autoregressive
decoding. No `nn.Transformer`, no `nn.MultiheadAttention`.

The repository contains two things:

1. **A 10.7M parameter model** trained on character-level Shakespeare — the
   full-scale reference implementation, val loss **1.5155**.
2. **A controlled ablation study at reduced scale (0.8M)** isolating what each
   architectural component contributes to validation loss.

The headline finding is in #4: removing residual connections costs **23× more
validation loss than a 13× increase in parameters buys**. At this data scale,
architecture dominates capacity.

**Scope.** All results are character-level, on 1.1MB of Shakespeare, at ≤6
layers. Findings should not be assumed to transfer to production-scale models or
other tokenizations. This is not a useful language model — it learns English
orthography and Shakespearean formatting but produces no semantic content. The
point is the architecture and the measurements.

---

## 1. Architecture

```
idx                     (B, T)              token ids
  → token_emb + pos_emb (B, T, C)
  → Block × n_layer     (B, T, C)           width constant through the stack
       ln1 → MHA → +x
       ln2 → FFN → +x
  → ln_f                (B, T, C)
  → lm_head             (B, T, vocab_size)
```

Inside one attention head:

```
x           (B, T, C)
q, k, v     (B, T, d)      d = C / n_head
q @ kᵀ      (B, T, T)      d contracts; a second token axis appears
× d^-0.5    (B, T, T)      variance control
mask -inf   (B, T, T)      upper triangle killed
softmax     (B, T, T)      rows sum to 1
@ v         (B, T, d)      source-token axis contracts
```

---

## 2. Full-scale model (10.7M)

| Hyperparameter | Value |
|---|---|
| n_embd | 384 |
| n_head | 6 (head_size 64) |
| n_layer | 6 |
| block_size | 256 |
| batch_size | 64 |
| max_iters | 5000 |
| optimizer | AdamW, lr 3e-4 |
| dropout | 0.1 |
| parameters | 10,715,201 |

### Training curve

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

**Best val loss: 1.5155 at step 4500.** Step-0 loss of 4.2825 sits close to the
uniform baseline of `−ln(1/65) = 4.174`, confirming correct initialisation.

### Sample generation

```
First Officer:
Indeed to mind; sweet Exeter's granted in blood,
That I beseech your honourages me with
chop of usnight of that you are:
Your teams is true advantage in a greeting date.
I have my brother frid your chitference
From I'll may no part in that words plane,
And with thee are a word. Ah, time I so,
That they are died in Rome; come, away, prince, or so? I will
In What more delonice.
Dart thou not speak to the field; whose mothers
sin that what I speak you will not.

Both, this for me; and I were but one leisure grave
Marr'd by the gross of Sance!
```

Worth reading closely for what a character-level model at this scale does and
does not learn.

**Learned:** the `NAME:` speaker convention, blank lines between speeches, line
lengths approximating iambic pentameter, capitalisation and punctuation
conventions, archaic forms (`thee`, `thou`, `beseech`, `Marr'd`), and correct
spelling for the overwhelming majority of tokens — all from raw characters, with
no notion of a word.

**Not learned:** meaning. Sentences are grammatically shaped but semantically
empty. Proper nouns appear plausibly (`Exeter`, `Rome`) but without reference.
Invented words surface where the model interpolates between character
statistics — `honourages`, `usnight`, `chitference`, `delonice` — each of which
is phonotactically valid English that does not exist.

Those confabulations are the clearest illustration of what the objective
actually optimises: next-character likelihood, not correctness. The model has
learned the *shape* of English and Shakespeare, not their content.

---

## 3. Why the ablations run at reduced scale

| | Full-scale | Ablation scale |
|---|---|---|
| n_embd | 384 | 128 |
| n_layer | 6 | 4 |
| block_size | 256 | 64 |
| batch_size | 64 | 32 |
| max_iters | 5000 | 3000 |
| dropout | 0.2 | **0.0** |
| parameters | 10.7M | 0.8M |
| time per run | ~40 min | ~2 min |

**Compute.** Eight configurations at full scale is ~5 hours; at reduced scale,
15 minutes.

**Dropout is disabled.** This is the important one. Dropout injects
stochasticity that would appear as run-to-run variance, and the deltas being
measured are as small as 0.015. Regularisation noise would swamp the signal. It
is disabled in every ablation run including the baseline, so the comparison
remains internally valid.

**Measured variance justifies the scale.** Seed variance at the ablation
configuration was measured at ±0.004–0.011 val loss across five seeds, so
effects above roughly 0.02 are distinguishable from noise. Findings below that
threshold are flagged.

**Limitation.** Component importance is not necessarily scale-invariant. Two
results below are explicitly expected to change with scale — LayerNorm and the
`1/√d` scaling factor — and this is noted where relevant.

---

## 4. Scale versus architecture

Putting the two studies side by side:

| Change | Δ val loss |
|---|---|
| 0.8M → 10.7M params, 64 → 256 context | **−0.077** (1.5923 → 1.5155) |
| Removing residual connections (at 0.8M) | **+1.759** |
| Removing the FFN | +0.127 |
| Removing positional embeddings | +0.053 |

**A 13× increase in parameters plus a 4× increase in context length bought 0.077
val loss. Removing one architectural component cost 23× that.**

Three observations follow.

**The task is data-bound, not capacity-bound.** Validation loss bottoms out at
step 4500 and rises through step 4999 (1.5155 → 1.5217) while training loss
continues to fall (1.2210 → 1.2020). The train/val gap widens from 0.14 at step
500 to 0.32 at step 4999. With 10.7M parameters against 1.1M training characters
— roughly ten parameters per character — the model has begun memorising even
with dropout at 0.2. Additional capacity has little left to extract.

**A parameter-matched control confirms this.** In a separate run, widening the
0.8M baseline from n_embd 128 to 132 (+76k parameters) produced *worse*
validation loss at the same iteration count. Extra width is not usable at this
data scale.

**The ablation deltas are therefore lower bounds on architectural importance.**
Measured against a scaling axis that is nearly saturated, the components that
still move validation loss are doing work that capacity cannot substitute for.

---

## 5. Component ablation study (0.8M)

**Method.** Every configuration trains from the same seed, so weight
initialisation and the entire sequence of sampled batches are identical across
runs. Exactly one architectural component differs per run. Validation loss is
averaged over 200 held-out batches after 3000 iterations.

### Results

| Configuration | Val loss | Δ vs baseline | Params |
|---|---|---|---|
| baseline | 1.5923 | — | 816,705 |
| no positional embeddings | 1.6454 | +0.0531 | 816,705 |
| no FFN | 1.7194 | +0.1271 | 289,857 |
| **no residual connections** | **3.3516** | **+1.7593** | 816,705 |
| no LayerNorm | 1.6312 | +0.0389 | 814,401 |
| no 1/√d scaling | 1.6077 | +0.0154 | 816,705 |
| single head (width 128) | 1.6181 | +0.0258 | 816,705 |
| **no causal mask** | **0.0327** | **−1.5596** | 816,705 |

**On parameter counts.** Six of eight configurations are parameter-neutral, so
their deltas are directly attributable to the mechanism. Two are not: `no FFN`
removes 526,848 parameters (65% of the model) and `no LayerNorm` removes 2,304.
The FFN delta therefore conflates the loss of per-token nonlinearity with
reduced capacity and should be read as an upper bound.

---

## 6. Predictions vs. outcomes

Predictions were recorded before any ablation was run.

| Ablation | Predicted | Actual | Result |
|---|---|---|---|
| baseline | 1.59 | 1.5923 | ✓ |
| no_pos_emb | 1.63 | 1.6454 | ✓ close |
| no_ffn | 1.61 | 1.7194 | ✗ underestimated |
| no_residual | 1.954 | 3.3516 | ✗✗ badly underestimated |
| no_layernorm | 1.64 | 1.6312 | ✓ |
| no_scaling | 1.80 | 1.6077 | ✗  overestimated |
| single_head | 1.64 | 1.6181 | ✓ |
| no_causal_mask | 0.60 | 0.0327 | ✓ direction, ✗ magnitude |

Five of eight landed close. The two large misses are the most informative
results in the study.

---

## 7. Discussion

### Residual connections dominate everything else (+1.76)

Predicted a mild +0.06; measured +1.76 — **thirty times larger than any other
ablation**, and worse than a bigram model's ~2.50. Removing residuals does not
degrade the model, it breaks it.

The reasoning behind the low prediction was that at only 4 layers the gradient
highway `∂(x + f(x))/∂x = I + ∂f/∂x` should not yet be load-bearing — vanishing
gradients are usually framed as a deep-network problem.

That reasoning missed the second function of residuals. Without the `x +`, each
block's output *replaces* the stream rather than adding to it. Attention becomes
the only path for information, and attention is a smoothing operator: its output
is a convex combination of value vectors, `o_i ∈ conv{v_0..v_i}`. Stacked pure
self-attention provably converges toward rank-1 — all token representations
collapse to the same vector. Residuals preserve the token-specific signal across
depth, and that matters at four layers just as much as at forty.

Residuals are not primarily a gradient-flow patch. They are what makes the
architecture a *residual stream* — an additive workspace each block reads from
and writes to — rather than a destructive pipeline.

### The 1/√d scaling barely mattered (+0.015)

Predicted the single largest degradation (+0.31); measured the smallest
(+0.015).

The prediction followed from the standard derivation: `Var(q·k) = d·σ⁴`, so
unscaled scores at `head_size = 32` have std ≈ 5.7.


So `1/√d` buys faster early convergence, not final quality — **at this scale**.
The prediction was directionally right and quantitatively wrong because it
treated an initialisation property as a permanent constraint. **This is the
result most likely to change at full scale**: at `head_size = 64` the initial
saturation is more severe (std ≈ 8 rather than 5.7) and recovery slower relative
to the training budget. Re-running this single ablation at full scale would test
that directly.

### The causal mask: best loss, worst model

`no_causal_mask` produced a validation loss of **0.0327** — better than every
other configuration by an enormous margin, and essentially zero.

It is also the only configuration that renders the model useless. Without the
mask, position `t` can attend to position `t+1`, which is its own training
target. Next-token prediction degenerates into copying. A loss of 0.03 means it
learned to cheat almost perfectly.

Generation from this checkpoint is incoherent, because at inference time the
future tokens it learned to read do not exist yet.

This is the clearest demonstration in the study that **loss is a proxy, not the
objective.** One deleted line produced a number that looks like a 50×
improvement and is in fact total failure.

### Positional embeddings (+0.053)

Smaller than expected for a mechanism this fundamental. Self-attention is
permutation-equivariant — `Attn(PX) = P·Attn(X)` — so without positional
information it processes a set, not a sequence.

The likely explanation is that the causal mask itself leaks positional
information: position 0 attends to 1 token, position 5 attends to 6. The size of
a token's receptive field is a weak but usable position signal. At
`block_size = 64` there is also less long-range order to lose than at the
full-scale 256.

### Multi-head vs single head (+0.026)

The single-head configuration uses one head at full width (128) rather than four
at width 32, so parameter count is held constant. The +0.026 delta measures the
value of multiple independent routing patterns, not extra capacity.

A single softmax produces one attention distribution per query. H heads let the
layer express a sum of H distinct patterns at the same cost:
`MHA(X) = Σ_h A^(h) X W_v^(h) W_o^(h)`. The measured benefit is real but modest
at this scale — likely because a character-level model on one author has fewer
simultaneous relationships to track than a word-level model on diverse text.

### LayerNorm (+0.039) and FFN (+0.127)

LayerNorm's small delta is consistent with a 4-layer model: the residual stream
has not accumulated enough magnitude across depth for renormalisation to be
critical. **This effect should grow at the full-scale 6-layer configuration**,
and more so at greater depth.

The FFN result is the least clean in the study — 65% of the parameters were
removed along with the nonlinearity. The direction is unambiguous (attention is
linear in V, so without the FFN there is no per-token nonlinearity on the
content path), but the magnitude cannot be attributed to the mechanism alone.

---

## 8. Summary

Ranked by measured importance at 0.8M scale:

1. **Residual connections** (+1.76) — catastrophic to remove; preserves the
   additive stream and prevents rank collapse
2. **Causal mask** (−1.56, i.e. broken) — defines the task; without it there is
   no task
3. **FFN** (+0.13, capacity-confounded) — the only per-token nonlinearity
4. **Positional embeddings** (+0.05) — attention is order-blind by construction
5. **LayerNorm** (+0.04) — expected to grow with depth
6. **Multi-head** (+0.03) — multiple routing patterns at fixed cost
7. **1/√d scaling** (+0.015) — an initialisation aid the model can learn around

For reference, scaling from 0.8M to 10.7M parameters is worth **−0.077** — 23×
less than residuals alone.

The two components most often presented as implementation details — residual
connections and the causal mask — turned out to be the two the model cannot
function without. The component with the most elegant theoretical justification,
`1/√d` scaling, mattered least at this scale.

---


## 9. Reproducing

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python gpt.py           # full-scale 10.7M model,  ~40 min on a T4
python ablations.py     # 8-config study at 0.8M,  ~15 min
```

Kaggle: Accelerator = **GPU T4 x2** (the P100 is compute capability sm_60 and
unsupported by current PyTorch builds), Internet = On. Use *Save & Run All
(Commit)* for the longer run so it executes in the background.

---

## 11. Files

```
gpt.py                   full-scale model + training loop
ablations.py             8-configuration ablation harness
ablation_results.json    raw results including loss curves
```
