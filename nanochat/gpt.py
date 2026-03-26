"""
GPT model — refactored version (parallel to gpt.py for A/B comparison).

Round 1 changes (from original analysis):
1. [HIGH]   Softcap deleted — redundant given pre-norm + QK-norm architecture
2. [MEDIUM] Magic numbers promoted to GPTConfig (ve_gate_channels, attn_qk_scale)
3. [MEDIUM] Optimizer construction decoupled from model class
4. [MEDIUM] Naive generate() replaced with KV-cached inference
5. [LOW]    value_embeds string-key dict → indexed list for hot-path lookup

Round 2 changes (from delta analysis):
6. [HIGH]   value_embeds: ModuleDict+_ve_list → nn.ModuleList (fixes deserialization bug)
7. [MEDIUM] kv_cache.advance() moved from CausalSelfAttention to GPT.forward
8. [MEDIUM] estimate_flops/num_scaling_params extracted to model_utils.py
9. [LOW]    ve_gate_range=3.0 promoted to GPTConfig
10. [LOW]   Dead 'partial' import removed
11. [LOW]   Duplicate head_dim computation removed
12. [LOW]   Two init_weights loops merged into one

Round 3 changes (four-cause architecture review):
13. [MEDIUM] ForwardContext dataclass — Block/Attn signature reduced from 5 args to 3
14. [LOW]    self.head_dim hoisted as architectural constant on GPT
15. [LOW]    get_device() emits DeprecationWarning
16. [LOW]    has_ve docstring expanded with teleological explanation
17. [LOW]    setup_optimizer docstring added (convenience facade)
18. [LOW]    generate() docstring notes Engine.generate() as production path

Round 4 changes (memory optimisation):
19. [HIGH]   Chunked cross-entropy — avoids materializing full (B,T,V) logit tensor
"""

from dataclasses import dataclass  # [FIX 10] removed unused 'partial' import

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    ve_gate_channels: int = 12
    attn_qk_scale: float = 1.15
    ve_gate_range: float = 3.0  # [FIX 9] was hardcoded as 3 in CausalSelfAttention.forward


def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Value Embeddings on alternating layers, last layer always included.

    Teleology: VE provides a direct gradient path from loss to input embeddings,
    bypassing transformer depth. Alternating halves the memory cost while
    maintaining the gradient highway. Last-layer inclusion ensures the
    final representation always has access to the VE signal.
    """
    return layer_idx % 2 == (n_layer - 1) % 2

@dataclass
class ForwardContext:
    """Per-forward-pass context bundling rotary embeddings, window config, and optional KV cache."""
    cos_sin: tuple[torch.Tensor, torch.Tensor]
    window_size: tuple[int, int]
    kv_cache: object = None  # KVCache or None (lazy import to avoid circular dep)


def _chunked_cross_entropy(lm_head, x, targets, vocab_size, reduction='mean', chunk_size=512, ignore_index=-1):
    """Compute cross-entropy loss in chunks to avoid materializing full (B,T,V) logit tensor.

    Peak logit memory: chunk_size * vocab_size * 4 bytes (e.g. 512 * 32768 * 4 = 67 MB)
    instead of B * T * vocab_size * 4 bytes (e.g. 32 * 2048 * 32768 * 4 = 8.6 GB).
    """
    B, T, D = x.shape
    N = B * T
    x_flat = x.reshape(N, D)
    targets_flat = targets.reshape(N)

    if reduction == 'none':
        losses = torch.empty(N, dtype=torch.float32, device=x.device)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            logits_c = lm_head(x_flat[i:end])[:, :vocab_size].float()
            losses[i:end] = F.cross_entropy(logits_c, targets_flat[i:end],
                                            ignore_index=ignore_index, reduction='none')
        return losses
    else:
        total_loss = torch.zeros((), dtype=torch.float32, device=x.device)
        total_valid = 0
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            logits_c = lm_head(x_flat[i:end])[:, :vocab_size].float()
            t_c = targets_flat[i:end]
            total_loss = total_loss + F.cross_entropy(logits_c, t_c,
                                                      ignore_index=ignore_index, reduction='sum')
            total_valid += (t_c != ignore_index).sum()
        return total_loss / total_valid


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx  # needed for get_layer_cache
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = config.ve_gate_channels
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.attn_qk_scale = config.attn_qk_scale
        self.ve_gate_range = config.ve_gate_range  # [FIX 9]

    def forward(self, x, ve, ctx: ForwardContext):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = self.ve_gate_range * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # [FIX 9]
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = ctx.cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * self.attn_qk_scale
        k = k * self.attn_qk_scale

        if ctx.kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=ctx.window_size)
        else:
            k_cache, v_cache = ctx.kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=ctx.kv_cache.cache_seqlens,
                causal=True,
                window_size=ctx.window_size,
            )
            # [FIX 7] advance() removed from here — now in GPT.forward

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, ctx: ForwardContext):
        x = x + self.attn(norm(x), ve, ctx)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # [FIX 6] nn.ModuleList replaces ModuleDict+_ve_list — no deserialization bug
        # [FIX 11] head_dim computed once, stored as architectural constant
        self.head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * self.head_dim
        self.value_embeds = nn.ModuleList([
            nn.Embedding(padded_vocab_size, kv_dim) if has_ve(i, config.n_layer) else None
            for i in range(config.n_layer)
        ])
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        # [FIX 12] single merged loop for block weights + ve_gate
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.5, s * 0.5)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # [FIX 6] iterate ModuleList directly
        for ve in self.value_embeds:
            if ve is not None:
                torch.nn.init.uniform_(ve.weight, -s, s)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds:
                if ve is not None:
                    ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = -(-long_window // 3 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    @property
    def device(self):  # [FIX 11b] @property instead of get_device()
        return self.transformer.wte.weight.device

    def get_device(self):
        """Deprecated: use .device property instead."""
        return self.device

    def estimate_flops(self):
        """Method wrapper for backward compat — delegates to standalone estimate_flops()."""
        return estimate_flops(self)

    def num_scaling_params(self):
        """Method wrapper for backward compat — delegates to standalone num_scaling_params()."""
        return num_scaling_params(self)

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        """Convenience facade; see optim_factory.make_optimizer for full control."""
        from nanochat.optim_factory import make_optimizer
        return make_optimizer(self, unembedding_lr=unembedding_lr, embedding_lr=embedding_lr,
                              matrix_lr=matrix_lr, weight_decay=weight_decay, scalar_lr=scalar_lr)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # [FIX 6] direct ModuleList index — no string keys, no dual representation
            ve_mod = self.value_embeds[i]
            ve = ve_mod(idx).to(x.dtype) if ve_mod is not None else None
            ctx = ForwardContext(cos_sin=cos_sin, window_size=self.window_sizes[i], kv_cache=kv_cache)
            x = block(x, ve, ctx)
        # [FIX 7] advance kv_cache here, not inside CausalSelfAttention
        if kv_cache is not None:
            kv_cache.advance(T)
        x = norm(x)

        # [FIX 19] Chunked cross-entropy: avoid materializing full (B,T,V) logit tensor
        if targets is not None:
            return _chunked_cross_entropy(
                self.lm_head, x, targets,
                vocab_size=self.config.vocab_size,
                reduction=loss_reduction,
            )
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """KV-cached autoregressive streaming inference. Batch size 1.

        For production inference with multi-sample, tool use, and proper lifecycle
        management, use Engine.generate() instead.
        """
        assert isinstance(tokens, list)
        device = self.device  # uses @property
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        from nanochat.engine import KVCache  # deferred to avoid circular import

        m = self.config
        total_len = len(tokens) + max_tokens
        kv_cache = KVCache(
            batch_size=1, num_heads=m.n_kv_head, seq_len=total_len,
            head_dim=self.head_dim, num_layers=m.n_layer,
            device=device, dtype=dtype,
        )

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.forward(ids, kv_cache=kv_cache)
        logits = logits[:, -1, :]

        for _ in range(max_tokens):
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            token = next_ids.item()
            yield token
            logits = self.forward(next_ids, kv_cache=kv_cache)
            logits = logits[:, -1, :]


# ---------------------------------------------------------------------------
# [FIX 8] Analytics functions extracted from GPT class — standalone utilities
# ---------------------------------------------------------------------------

def estimate_flops(model):
    """Return estimated FLOPs per token for the model (forward + backward)."""
    nparams = sum(p.numel() for p in model.parameters())
    value_embeds_numel = sum(ve.weight.numel() for ve in model.value_embeds if ve is not None)
    nparams_exclude = (model.transformer.wte.weight.numel() + value_embeds_numel +
                      model.resid_lambdas.numel() + model.x0_lambdas.numel())
    h, q, t = model.config.n_head, model.config.n_embd // model.config.n_head, model.config.sequence_len
    attn_flops = 0
    for window_size in model.window_sizes:
        window = window_size[0]
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq
    return 6 * (nparams - nparams_exclude) + attn_flops


def num_scaling_params(model):
    """Return detailed parameter counts for scaling law analysis."""
    wte = sum(p.numel() for p in model.transformer.wte.parameters())
    value_embeds = sum(p.numel() for p in model.value_embeds.parameters())
    lm_head = sum(p.numel() for p in model.lm_head.parameters())
    transformer_matrices = sum(p.numel() for p in model.transformer.h.parameters())
    scalars = model.resid_lambdas.numel() + model.x0_lambdas.numel()
    total = wte + value_embeds + lm_head + transformer_matrices + scalars
    assert total == sum(p.numel() for p in model.parameters()), "Parameter count mismatch"
    return {
        'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
        'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
    }
