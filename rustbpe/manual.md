# rustbpe — User Manual

A fast BPE tokenizer for macOS Apple Silicon, written in Rust with Python bindings.
Built for the nanochat project. Trains on your text, exports to tiktoken for inference.

---

## What is this, in plain language?

### The problem rustbpe solves

Before a language model can read text, the text has to be converted into numbers.
That conversion is called **tokenization**. The process of *learning* how to convert
text — deciding which letter groups become single tokens — is called **training a tokenizer**.

Training is slow in pure Python. rustbpe does it in Rust, using all your CPU cores in
parallel, so it finishes in seconds instead of minutes.

### What is BPE?

**Byte Pair Encoding (BPE)** is the tokenization algorithm used by GPT-2, GPT-4, LLaMA,
and most modern language models. Here is the intuition:

1. Start with every individual byte (256 symbols: a, b, c … space, punctuation, etc.)
2. Find the two symbols that appear side-by-side most often in your text
3. Merge them into a new single symbol (e.g. `t` + `h` → `th`)
4. Repeat until you have the vocabulary size you want (e.g. 4,096 tokens)

The result is a vocabulary where common words become single tokens (`the`, `ing`,
`tion`) and rare words are split into pieces (`un` + `usual`). Shorter token sequences
mean faster, cheaper model training.

### What rustbpe does specifically

| Step | What happens |
|---|---|
| **Train** | Reads your text corpus through a Python iterator, splits it using a GPT-4 style regex, counts byte-pair frequencies in parallel using Rayon (Rust's data-parallelism library), then runs the BPE merge loop to produce a vocabulary |
| **Export** | `get_mergeable_ranks()` returns the vocabulary in the format tiktoken expects |
| **Encode** | `encode()` converts a string to token IDs using the learned vocabulary |

In normal nanochat usage you train with rustbpe, export to tiktoken, and use tiktoken
for all inference encoding. The Rust `encode()` is available for testing and offline
token counting.

---

## Prerequisites

### Required once

```bash
# 1. Rust toolchain (installs rustc + cargo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. maturin — the tool that compiles Rust into a Python extension
pip install maturin
# or, inside the nanochat uv environment:
uv add --dev maturin

# 3. Python 3.10 (nanochat pins this version)
pyenv install 3.10
pyenv local 3.10        # run this in the nanochat-master/ directory
```

### What you already have (no action needed)

- The `rustbpe/` directory with `Cargo.toml`, `pyproject.toml`, and `src/lib.rs`
  is already in the repo.
- `pyproject.toml` at the repo root already lists `rustbpe>=0.1.0` as a dependency
  and points uv at the local source via `rustbpe = { path = "rustbpe/" }`.

---

## Building

### Development build (fast iteration, debug-friendly)

Run this once — or again whenever you change `src/lib.rs`:

```bash
cd rustbpe
maturin develop --release --features extension-module
cd ..
```

`--release` turns on full Rust optimizations (LTO, single codegen unit). Without it
the extension runs ~10× slower. `--features extension-module` tells pyo3 it is being
compiled as a Python `.so` file rather than a standalone Rust binary.

After this, `import rustbpe` works in any Python process in the same virtual environment.

### Verify the build

```bash
python -c "import rustbpe; t = rustbpe.Tokenizer(); print('rustbpe ok:', t)"
```

Expected output:
```
rustbpe ok: <rustbpe.Tokenizer object at 0x...>
```

### Smoke-test the full training pipeline

```bash
python scripts/tok_train.py --max-chars 500000 --vocab-size 1000
```

This trains a 1,000-token vocabulary on 500 KB of your corpus. It should complete in
a few seconds on Apple Silicon.

---

## Python API

### `rustbpe.Tokenizer()`

Creates a new, empty tokenizer. No arguments.

```python
import rustbpe
tok = rustbpe.Tokenizer()
```

---

### `tok.train_from_iterator(iterator, vocab_size, buffer_size=8192, pattern=None)`

Trains the tokenizer on text from a Python iterator.

| Parameter | Type | Description |
|---|---|---|
| `iterator` | any iterable of `str` | Your text corpus. Can be a generator, HuggingFace dataset iterator, file lines — anything Python can iterate over |
| `vocab_size` | `int` | Final vocabulary size. Must be ≥ 256. Common values: 1,024 / 4,096 / 32,768 / 50,257 |
| `buffer_size` | `int` | How many strings to pull from the iterator at once before releasing Python's GIL for parallel processing. Default 8,192 is good for most uses |
| `pattern` | `str \| None` | Custom regex for splitting text into pieces before BPE. `None` uses the GPT-4 default pattern |

```python
# From a list
texts = ["Hello world", "The quick brown fox", ...]
tok.train_from_iterator(texts, vocab_size=4096)

# From a HuggingFace dataset (streaming)
from datasets import load_dataset
ds = load_dataset("openwebtext", split="train", streaming=True)
tok.train_from_iterator((row["text"] for row in ds), vocab_size=32768)
```

**What happens under the hood:**
The iterator is drained in batches. Each batch is processed in parallel by Rayon across
all CPU cores: each string is split by the GPT-4 regex and the pieces are counted.
After all batches, the BPE merge loop runs to completion.

---

### `tok.get_mergeable_ranks()`

Returns the learned vocabulary in tiktoken format.

```python
ranks = tok.get_mergeable_ranks()
# returns: list of (bytes_object, token_id) tuples
# e.g. [(b'a', 97), (b'b', 98), ..., (b'the', 1024), ...]
```

Use this to register your tokenizer with tiktoken:

```python
import tiktoken

ranks = tok.get_mergeable_ranks()
mergeable_ranks = {token_bytes: rank for token_bytes, rank in ranks}

enc = tiktoken.Encoding(
    name="my_tokenizer",
    pat_str=tok.get_pattern(),
    mergeable_ranks=mergeable_ranks,
    special_tokens={},
)

# Now use tiktoken for all encoding/decoding
ids = enc.encode("Hello, world!")
text = enc.decode(ids)
```

---

### `tok.encode(text)`

Encodes a string directly using the learned merges. Returns a list of token IDs.

```python
ids = tok.encode("The quick brown fox")
# e.g. [464, 2068, 7586, 21831]
```

**Note:** In normal nanochat usage, tiktoken handles encoding at inference time.
`encode()` is useful for testing, token counting, and offline preprocessing.

---

### `tok.get_pattern()`

Returns the regex pattern string the tokenizer was trained with.

```python
pattern = tok.get_pattern()
print(pattern)
# "'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|..."
```

---

## Complete usage example

```python
import rustbpe
import tiktoken

# 1. Train
tok = rustbpe.Tokenizer()
corpus = ["Hello world!", "The fox jumped.", "BPE tokenization rocks."]
tok.train_from_iterator(corpus, vocab_size=512)

# 2. Export to tiktoken
mergeable_ranks = {b: r for b, r in tok.get_mergeable_ranks()}
enc = tiktoken.Encoding(
    name="demo",
    pat_str=tok.get_pattern(),
    mergeable_ranks=mergeable_ranks,
    special_tokens={},
)

# 3. Encode / decode
ids = enc.encode("Hello world!")
print(ids)                    # [72, 101, 108, ...]
print(enc.decode(ids))        # "Hello world!"
```

---

## Building distribution wheels (macOS)

If you want to share the built extension without requiring others to compile from source:

```bash
# Apple Silicon wheel
maturin build --release --features extension-module --target aarch64-apple-darwin

# Intel Mac wheel
maturin build --release --features extension-module --target x86_64-apple-darwin
```

Wheels are written to `rustbpe/target/wheels/`. Install with:

```bash
pip install rustbpe/target/wheels/rustbpe-0.1.0-*.whl
```

---

## What the fixes mean (plain language)

The original code had several bugs and performance problems. Here is what was fixed and why it matters to you:

### Stability fix: counting numbers that were too small

**The bug:** The training algorithm counted how often pairs of tokens appeared using
a 32-bit integer. On a large corpus (millions of pairs), this counter could silently
overflow and wrap around to a garbage value, causing the wrong merges to be chosen.
The counter was also compared with a 64-bit number in a way that could produce a
nonsense result if the count ever went negative.

**The fix:** All counters are now 64-bit. Negative values are clamped to zero before
any comparison. A large corpus will never corrupt training silently.

**What it means for you:** Training results are now trustworthy on real-world data.

---

### Reproducibility fix: random ordering

**The bug:** The order in which the program processed internal data structures
depended on the randomness of Rust's hash maps. Two training runs on the same corpus
could produce slightly different vocabularies when token frequencies were tied.

**The fix:** All positions and pairs are sorted before use. The input corpus is
materialized in alphabetical order. Tie-breaks are explicit and deterministic.

**What it means for you:** Same corpus + same vocab_size = identical tokenizer, every
time. This is essential for comparing experiments or sharing tokenizers with teammates.

---

### Performance fix: heap memory bloat

**The bug:** Each entry in the priority queue (the data structure that picks which pair
to merge next) carried a large set of word positions inside it. As training progressed,
these sets grew and were copied every time an entry moved in the queue — thousands of
times per merge step.

**The fix:** Positions are stored in a single side map outside the queue. Each queue
entry is now just two numbers (a pair and a count). Memory usage is dramatically lower
and the queue operates much faster.

**What it means for you:** Training is significantly faster, especially for large
vocabularies or large corpora, and uses much less RAM.

---

### Performance fix: skipping unnecessary work

**The bug:** When merging a pair of tokens, the algorithm would re-scan every word that
*might* contain the pair — including words that had already had it merged away in a
previous step. Each scan was wasted work.

**The fix:** Before scanning a word, a quick check verifies that the pair actually
exists there. Words that no longer contain the pair are skipped immediately.

**What it means for you:** Each merge step is faster, and the speedup grows as training
progresses (there are more and more stale entries as merges accumulate).

---

### Performance fix: slow text encoding

**The bug:** The `encode()` function worked by repeatedly scanning all token pairs,
finding the best one, merging it, and shifting the entire array — an O(n²) algorithm.
For a 1,000-byte input, that could mean 500,000 operations.

**The fix:** Encoding now uses a priority queue and a linked list so each merge
requires only O(log n) work — the same approach used by production tokenizers.

**What it means for you:** Encoding long strings is orders of magnitude faster. This
matters most for batch preprocessing of large datasets.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'rustbpe'`

The extension has not been built yet, or was built for a different Python environment.

```bash
cd rustbpe
maturin develop --release --features extension-module
```

### `maturin: command not found`

```bash
pip install maturin
# or
uv add --dev maturin
```

### Rust compiler errors after editing `src/lib.rs`

Run `cargo check` first for faster feedback:

```bash
cd rustbpe
cargo check
```

### Build is very slow

Make sure you pass `--release`. Without it, Rust compiles without optimizations and
the resulting extension is much slower.

### `uv sync` fails after adding rustbpe as a local dependency

The lockfile needs regenerating:

```bash
uv lock
uv sync --extra cpu
```

---

## Project structure

```
rustbpe/
├── Cargo.toml          # Rust package definition and dependencies
├── pyproject.toml      # Python build system config (maturin)
├── .gitignore          # Excludes the large Rust build cache (target/)
├── README.md           # This file
└── src/
    └── lib.rs          # All Rust source code (~575 lines)
```

The root `pyproject.toml` points uv at this directory:
```toml
[tool.uv.sources]
rustbpe = { path = "rustbpe/" }
```

This means `uv sync` builds from local source rather than downloading from PyPI,
so your local fixes are always what gets installed.
