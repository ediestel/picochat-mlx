#!/usr/bin/env python3
"""
Performance & correctness test suite for picochat gpt.py (refactored v2).

Mirrors the nanochat compare_chunked_ce.py and compare_refactor.py benchmarks.
Run: uv run python tests/test_performance.py
"""

import os
os.environ["NANOCHAT_DTYPE"] = "float32"

import time
import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
import nanochat.gpt as gpt_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(seed=42):
    torch.manual_seed(seed)
    cfg = gpt_mod.GPTConfig(
        sequence_len=128, vocab_size=256, n_layer=4,
        n_head=4, n_kv_head=4, n_embd=64, window_pattern="SL",
    )
    with torch.device("meta"):
        model = gpt_mod.GPT(cfg)
    model.to_empty(device=torch.device("cpu"))
    torch.manual_seed(seed)
    model.init_weights()
    model.eval()
    return model


def random_batch(vocab_size=256, batch=2, seq_len=32, seed=123):
    torch.manual_seed(seed)
    idx = torch.randint(0, vocab_size, (batch, seq_len))
    targets = torch.randint(0, vocab_size, (batch, seq_len))
    return idx, targets


def time_forward(model, idx, targets, warmup=5, repeat=50):
    for _ in range(warmup):
        model(idx, targets=targets)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        model(idx, targets=targets)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    trim = max(1, len(times) // 10)
    return sum(times[trim:-trim]) / len(times[trim:-trim])


def time_generate(model, tokens, max_tokens=32, warmup=2, repeat=10):
    for _ in range(warmup):
        list(model.generate(tokens, max_tokens=max_tokens, temperature=0))
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        list(model.generate(tokens, max_tokens=max_tokens, temperature=0))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    trim = max(1, len(times) // 10)
    return sum(times[trim:-trim]) / len(times[trim:-trim])


def print_table(title, headers, rows):
    col_widths = []
    for i, h in enumerate(headers):
        w = len(h)
        for row in rows:
            if i < len(row):
                w = max(w, len(str(row[i])))
        col_widths.append(w + 2)
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    def fmt_row(cells):
        parts = [str(c).center(w) for c, w in zip(cells, col_widths)]
        return "|" + "|".join(parts) + "|"
    print(f"\n{'=' * len(sep)}")
    print(f" {title}")
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)
    print()


# ===========================================================================
# TEST 1: Parameter count & structural checks
# ===========================================================================

def test_structure():
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())

    from nanochat.gpt import has_ve
    ve_count = sum(1 for ve in model.value_embeds if ve is not None)
    ve_none = sum(1 for ve in model.value_embeds if ve is None)

    rows = [
        ("Parameter count", f"{n_params:,}"),
        ("value_embeds type", type(model.value_embeds).__name__),
        ("VE layers (active)", str(ve_count)),
        ("VE layers (None)", str(ve_none)),
        ("head_dim stored", str(model.head_dim)),
        ("GPTConfig.ve_gate_channels", str(model.config.ve_gate_channels)),
        ("GPTConfig.attn_qk_scale", str(model.config.attn_qk_scale)),
        ("GPTConfig.ve_gate_range", str(model.config.ve_gate_range)),
        ("device property works", str(model.device == model.get_device())),
        ("estimate_flops() works", str(model.estimate_flops() > 0)),
        ("num_scaling_params() works", str(model.num_scaling_params()['total'] == n_params)),
    ]
    return rows


# ===========================================================================
# TEST 2: Chunked CE — reduction='mean'
# ===========================================================================

def test_chunked_mean():
    model = build_model()
    idx, targets = random_batch()
    with torch.no_grad():
        # Chunked (model.forward with targets)
        loss_chunked = model(idx, targets=targets).item()
        # Standard (manual full logits)
        x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = gpt_mod.norm(x)
        x0 = x
        for i, block in enumerate(model.transformer.h):
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve_mod = model.value_embeds[i]
            ve = ve_mod(idx).to(x.dtype) if ve_mod is not None else None
            cos_sin = model.cos[:, :idx.size(1)], model.sin[:, :idx.size(1)]
            ctx = gpt_mod.ForwardContext(cos_sin=cos_sin, window_size=model.window_sizes[i])
            x = block(x, ve, ctx)
        x = gpt_mod.norm(x)
        logits = model.lm_head(x)[..., :model.config.vocab_size].float()
        loss_standard = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1).item()

    delta = abs(loss_chunked - loss_standard)
    return [
        ("loss (chunked CE)", f"{loss_chunked:.8f}"),
        ("loss (full logits)", f"{loss_standard:.8f}"),
        ("delta", f"{delta:.10f}"),
        ("match (< 1e-5)", str(delta < 1e-5)),
    ]


# ===========================================================================
# TEST 3: Chunked CE — reduction='none'
# ===========================================================================

def test_chunked_none():
    model = build_model()
    idx, targets = random_batch()
    with torch.no_grad():
        loss = model(idx, targets=targets, loss_reduction='none')
    return [
        ("output shape", str(tuple(loss.shape))),
        ("expected shape", str((idx.size(0) * idx.size(1),))),
        ("all finite", str(torch.isfinite(loss).all().item())),
        ("min loss", f"{loss.min().item():.4f}"),
        ("max loss", f"{loss.max().item():.4f}"),
    ]


# ===========================================================================
# TEST 4: Chunked CE — ignore_index
# ===========================================================================

def test_chunked_ignore():
    model = build_model()
    idx, targets = random_batch()
    torch.manual_seed(999)
    mask = torch.rand_like(targets.float()) < 0.3
    targets[mask] = -1
    n_masked = mask.sum().item()
    n_total = targets.numel()

    with torch.no_grad():
        loss = model(idx, targets=targets).item()

    return [
        ("masked tokens", f"{n_masked}/{n_total}"),
        ("loss", f"{loss:.8f}"),
        ("finite", str(torch.isfinite(torch.tensor(loss)).item())),
    ]


# ===========================================================================
# TEST 5: Chunked CE — partial chunk (B*T not divisible by 512)
# ===========================================================================

def test_chunked_partial():
    model = build_model()
    torch.manual_seed(456)
    idx = torch.randint(0, 256, (3, 100))  # B*T = 300
    targets = torch.randint(0, 256, (3, 100))
    with torch.no_grad():
        loss = model(idx, targets=targets).item()
    return [
        ("B*T", "300 (not divisible by 512)"),
        ("loss", f"{loss:.8f}"),
        ("finite", str(torch.isfinite(torch.tensor(loss)).item())),
    ]


# ===========================================================================
# TEST 6: Forward pass performance
# ===========================================================================

def test_forward_perf():
    model = build_model()
    configs = [(2, 32, "B=2  T=32"), (4, 64, "B=4  T=64"), (8, 128, "B=8  T=128")]
    rows = []
    for batch, seq, label in configs:
        idx, targets = random_batch(batch=batch, seq_len=seq)
        ms = time_forward(model, idx, targets)
        rows.append((label, f"{ms:.2f} ms"))
    return rows


# ===========================================================================
# TEST 7: Generate performance
# ===========================================================================

def test_generate_perf():
    model = build_model()
    tokens = list(range(16))  # 16 prompt tokens

    ms = time_generate(model, tokens, max_tokens=32)
    gen_tokens = list(model.generate(tokens, max_tokens=32, temperature=0))

    return [
        ("prompt tokens", "16"),
        ("generated tokens", "32"),
        ("time", f"{ms:.1f} ms"),
        ("tokens/sec", f"{32 / (ms / 1000):.0f}"),
        ("first 8 tokens", str(gen_tokens[:8])),
    ]


# ===========================================================================
# TEST 8: Generate correctness
# ===========================================================================

def test_generate_correctness():
    model = build_model()
    tokens = list(range(10))

    # Determinism at temperature=0
    gen1 = list(model.generate(tokens, max_tokens=10, temperature=0))
    gen2 = list(model.generate(tokens, max_tokens=10, temperature=0))
    deterministic = gen1 == gen2

    # Seed reproducibility at temperature=1
    gen_a = list(model.generate(tokens, max_tokens=10, temperature=1.0, seed=42))
    gen_b = list(model.generate(tokens, max_tokens=10, temperature=1.0, seed=42))
    reproducible = gen_a == gen_b

    # Different seeds produce different output
    gen_c = list(model.generate(tokens, max_tokens=10, temperature=1.0, seed=1))
    gen_d = list(model.generate(tokens, max_tokens=10, temperature=1.0, seed=999))
    varied = gen_c != gen_d

    return [
        ("temp=0 deterministic", str(deterministic)),
        ("same seed reproducible", str(reproducible)),
        ("diff seeds vary", str(varied)),
        ("temp=0 output", str(gen1[:8])),
    ]


# ===========================================================================
# TEST 9: Training convergence (50 steps)
# ===========================================================================

def test_training():
    n_steps = 50
    torch.manual_seed(42)
    cfg = gpt_mod.GPTConfig(
        sequence_len=64, vocab_size=256, n_layer=4,
        n_head=4, n_kv_head=4, n_embd=64, window_pattern="SL",
    )
    with torch.device("meta"):
        model = gpt_mod.GPT(cfg)
    model.to_empty(device=torch.device("cpu"))
    torch.manual_seed(42)
    model.init_weights()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    losses = []
    for step in range(n_steps):
        torch.manual_seed(step + 1000)
        idx = torch.randint(0, cfg.vocab_size, (4, 64))
        tgt = torch.randint(0, cfg.vocab_size, (4, 64))
        loss = model(idx, targets=tgt)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    rows = []
    for s in [0, 9, 24, 49]:
        rows.append((str(s + 1), f"{losses[s]:.6f}"))
    rows.append(("avg last 10", f"{sum(losses[-10:])/10:.6f}"))
    rows.append(("converging", str(losses[-1] < losses[0])))
    return rows


# ===========================================================================
# TEST 10: Stress test — scaled lm_head weights
# ===========================================================================

def test_stress():
    rows = []
    for scale in [1, 10, 100, 500]:
        model = build_model()
        with torch.no_grad():
            model.lm_head.weight.mul_(scale)
        idx, targets = random_batch()
        with torch.no_grad():
            logits = model(idx)
            max_logit = logits.abs().max().item()
            loss = model(idx, targets=targets).item()
            finite = torch.isfinite(torch.tensor(loss)).item()
        rows.append((f"x{scale}", f"{max_logit:.2f}", f"{loss:.4f}", str(finite)))
    return rows


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 72)
    print(" PICOCHAT gpt.py — Performance & Correctness Test Suite")
    print(f" Device: CPU | dtype: {COMPUTE_DTYPE}")
    print("=" * 72)

    print_table("TEST 1: Structural Checks",
                ["Check", "Result"],
                test_structure())

    print_table("TEST 2: Chunked CE — reduction='mean'",
                ["Check", "Result"],
                test_chunked_mean())

    print_table("TEST 3: Chunked CE — reduction='none'",
                ["Check", "Result"],
                test_chunked_none())

    print_table("TEST 4: Chunked CE — ignore_index masking",
                ["Check", "Result"],
                test_chunked_ignore())

    print_table("TEST 5: Chunked CE — partial chunk (B*T=300)",
                ["Check", "Result"],
                test_chunked_partial())

    print_table("TEST 6: Forward Pass Performance (trimmed mean, 50 runs)",
                ["Config", "Latency"],
                test_forward_perf())

    print_table("TEST 7: Generate Performance (KV-cached)",
                ["Metric", "Value"],
                test_generate_perf())

    print_table("TEST 8: Generate Correctness",
                ["Check", "Result"],
                test_generate_correctness())

    print_table("TEST 9: Training Convergence (50 steps, AdamW lr=3e-4)",
                ["Step", "Loss"],
                test_training())

    print_table("TEST 10: Stress Test — Scaled lm_head Weights",
                ["Scale", "max |logit|", "Loss", "Finite"],
                test_stress())

    print("All tests completed.")


if __name__ == "__main__":
    main()
