

import json
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# FIXED EXPERIMENTAL SETUP -- do not vary these between runs
# ----------------------------------------------------------------------------
batch_size    = 32
block_size    = 64      # bigger than 8 so positional info actually matters
max_iters     = 5000
eval_interval = 500
eval_iters    = 200
learning_rate = 1e-3
n_embd        = 128
n_head        = 4
n_layer       = 4
dropout       = 0.0
SEED          = 1337

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ----------------------------------------------------------------------------
# PARAMETERIZED MODEL
#
# cfg is a dict of booleans. Baseline = everything True.
# Each ablation flips exactly one to False.
# ----------------------------------------------------------------------------
DEFAULT_CFG = {
    'pos_emb':   True,   # add positional embeddings
    'ffn':       True,   # include the feed-forward sublayer
    'residual':  True,   # residual (skip) connections
    'layernorm': True,   # LayerNorm before each sublayer
    'scaling':   True,   # multiply scores by 1/sqrt(d_k)
    'causal':    True,   # causal mask (decoder vs encoder)
    'multihead': True,   # if False, collapse to a single head of width n_embd
}


class Head(nn.Module):
    def __init__(self, head_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)

        wei = q @ k.transpose(-2, -1)
        if self.cfg['scaling']:
            wei = wei * k.shape[-1] ** -0.5
        if self.cfg['causal']:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size, cfg):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, cfg) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # multihead ablation: 1 head at full width, same param count as n_head heads
        heads = n_head if cfg['multihead'] else 1
        head_size = n_embd // heads
        self.sa = MultiHead(heads, head_size, cfg)
        self.ffwd = FFN() if cfg['ffn'] else None
        self.ln1 = nn.LayerNorm(n_embd) if cfg['layernorm'] else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if cfg['layernorm'] else nn.Identity()

    def forward(self, x):
        if self.cfg['residual']:
            x = x + self.sa(self.ln1(x))
            if self.ffwd is not None:
                x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(self.ln1(x))
            if self.ffwd is not None:
                x = self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) if cfg['layernorm'] else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        if self.cfg['pos_emb']:
            x = x + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# ----------------------------------------------------------------------------
# RUNNER
# ----------------------------------------------------------------------------
def run(name, overrides):
    """Train one configuration and return its record."""
    cfg = {**DEFAULT_CFG, **overrides}

    # Same seed for every run: identical init and identical batch sequence,
    # so the only difference between runs is the architecture.
    torch.manual_seed(SEED)

    model = GPT(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    t0 = time.time()
    curve = []
    diverged = False

    for it in range(max_iters):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            curve.append({'iter': it, **losses})
            print(f"  [{name}] step {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
            if math.isnan(losses['val']) or math.isinf(losses['val']):
                diverged = True
                print(f"  [{name}] DIVERGED at step {it}")
                break

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    final = estimate_loss(model) if not diverged else {'train': float('nan'), 'val': float('nan')}
    elapsed = time.time() - t0

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(ctx, 300)[0].tolist()) if not diverged else "(diverged)"

    print(f"  [{name}] FINAL val {final['val']:.4f}  ({elapsed:.0f}s, {nparams/1e6:.3f}M params)\n")

    return {
        'name': name,
        'config': cfg,
        'params': nparams,
        'final_train': final['train'],
        'final_val': final['val'],
        'curve': curve,
        'seconds': elapsed,
        'diverged': diverged,
        'sample': sample,
    }


EXPERIMENTS = [
    ('baseline',        {}),
    ('no_pos_emb',      {'pos_emb': False}),
    ('no_ffn',          {'ffn': False}),
    ('no_residual',     {'residual': False}),
    ('no_layernorm',    {'layernorm': False}),
    ('no_scaling',      {'scaling': False}),
    ('single_head',     {'multihead': False}),
    ('no_causal_mask',  {'causal': False}),   # expect loss to COLLAPSE -- it's cheating
]

if __name__ == '__main__':
    results = []
    for name, overrides in EXPERIMENTS:
        print(f"=== {name} ===")
        results.append(run(name, overrides))

    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ---- summary table (markdown, paste straight into README) ----
    base = next(r for r in results if r['name'] == 'baseline')['final_val']

    print("\n\n| Configuration | Val loss | Δ vs baseline | Params |")
    print("|---|---|---|---|")
    for r in results:
        if r['diverged']:
            print(f"| {r['name']} | diverged | — | {r['params']:,} |")
        else:
            d = r['final_val'] - base
            ds = "—" if r['name'] == 'baseline' else f"{d:+.4f}"
            print(f"| {r['name']} | {r['final_val']:.4f} | {ds} | {r['params']:,} |")
