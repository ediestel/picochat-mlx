# rustbpe — Install & Fine-Tuning Guide (M4 Mac Mini 16 GB)

## Hardware Context (March 2026)

Running a **Mac Mini M4 with 16 GB unified memory** puts you in a solid position for local AI work using picochat / nanochat tools, especially MLX-based fine-tuning and inference on small-to-mid-size models. The M4's efficiency + unified memory architecture gives real headroom for practical domain adaptation on weight-loss journals and household stocking logs.

---

## Realistic Capabilities on M4 Mac Mini 16 GB

- **Inference** (running / chatting with models): Very comfortable for 3B–8B models at Q4/Q5 quantization (~4–6 GB loaded weights + KV cache for 4k–8k context). Expect 30–60+ tokens/sec depending on the model.
- **Fine-tuning / LoRA**: Feasible and reasonably fast for models up to ~7–8B params (e.g., Qwen2.5-7B, Llama-3.2-3B, Phi-4-mini 3.8B, Gemma-2 variants).
  - Typical run: 1k–5k steps on 5k–20k examples → 20–90 minutes.
  - Use continued pre-training on raw text logs (lower memory use) or LoRA/QLoRA for instruction-style fine-tuning.
  - Batch size 1–4, seq length 1024–2048, LoRA rank 16–32, gradient checkpointing enabled → stays within 12–15 GB peak usage.
- **Tokenizer training** with rustbpe: Still trivial (seconds–minutes, low RAM footprint).
- **Limitations**:
  - Full (non-LoRA) fine-tuning of 7B+ models → tight or impossible (needs ~3–4x model size for gradients/optimizers).
  - 13B+ models → inference only at very low quantization / short context, or heavy swapping.
  - Long contexts (>8k–16k tokens) or heavy multitasking during training → mild paging to SSD (still usable, but slower).

**Bottom line**: 16 GB is viable entry-level for this use case in 2026 — you can meaningfully adapt a model to know your calorie tracking style, pantry patterns, favorite low-cal recipes, shopping reminders, etc. It is not "pro" territory (24–32 GB would be smoother for 7B+ LoRA or longer runs), but it is far better than 8 GB.

---

## Recommended Models (Tailored for 16 GB M4)

Focus on **continued pre-training** first (easiest on memory), then add LoRA if needed for chat/instruction behavior.

| Priority | Model (Hugging Face path)           | Params | Quant Level | Why Good for Your Apps                  | Expected Fine-Tune Time (1k–3k steps) |
|----------|-------------------------------------|--------|-------------|-----------------------------------------|---------------------------------------|
| 1        | Qwen/Qwen2.5-3B-Instruct           | 3B     | Q5_K_M      | Strong small instruct model, great reasoning | 20–45 min                       |
| 2        | microsoft/Phi-4-mini-instruct      | 3.8B   | Q5          | Excellent for structured logs/meals     | 25–60 min                             |
| 3        | google/gemma-2-2b-it               | 2B     | Q6          | Fastest, clean outputs                  | 15–30 min                             |
| 4 (stretch) | meta-llama/Llama-3.2-3B-Instruct | 3B   | Q5          | Solid if you prefer the Llama format    | 30–70 min                             |

---

## Step-by-Step: Fine-Tune on Your Data (Using MLX)

### 1. Install / Update Tools

```bash
uv add mlx mlx-lm datasets accelerate peft
```

### 2. Prepare Your Dataset

Collect logs into text files or JSONL (aim for 1k–20k lines total).

**Simple continued pre-training format** (raw text, one entry per line):
```
2025-03-10 Breakfast: 2 eggs + spinach 320 kcal. Walked 45 min. Goal -500/day.
Pantry: rice 4kg (80% full), milk 1L left. Buy pasta next week.
User: Suggest dinner under 600 kcal with eggs and tomatoes.
```

**Chat-style format** (better results for instruction following):
```jsonl
{"text": "<|im_start|>user\nI logged 400 kcal lunch, walked 30 min. Suggest low-carb dinner.<|im_end|>\n<|im_start|>assistant\nGrilled chicken + veggies salad ~480 kcal. Use your remaining tomatoes...<|im_end|>\n"}
```

Save files in a `data/` folder.

### 3. Optional: Custom Tokenizer with rustbpe

Recommended for domain terms like "kcal", brand names, and measurement units.

```python
import rustbpe
tok = rustbpe.Tokenizer()

def iterator():
    import glob
    for file in glob.glob("data/*.txt") + glob.glob("data/*.jsonl"):
        with open(file) as f:
            yield from (line.strip() for line in f if line.strip())

tok.train_from_iterator(iterator(), vocab_size=8192)  # or 4096–16384
ranks = {b: r for b, r in tok.get_mergeable_ranks()}
# Save ranks/pattern for later use in MLX or tiktoken
```

### 4. Convert Model to MLX Format

Downloads ~2–4 GB once:

```bash
python -m mlx_lm.convert \
    --hf-path Qwen/Qwen2.5-3B-Instruct \
    --mlx-path ./mlx-qwen-3b \
    --q-bits 5
```

### 5. Run Continued Pre-Training

Safest approach on 16 GB:

```bash
python -m mlx_lm.lora \
    --model ./mlx-qwen-3b \
    --train \
    --data data/ \
    --iters 2000 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --adapter-path adapters/myhealthstock \
    --save-every 500 \
    --max-seq-length 1536 \
    --grad-checkpoint
```

Monitor Activity Monitor — if it hits 15+ GB, drop `--batch-size` or `--max-seq-length`.

### 6. Merge & Quantize for Fast App Use

```bash
python -m mlx_lm.fuse \
    --model ./mlx-qwen-3b \
    --adapter-path adapters/myhealthstock/final

python -m mlx_lm.convert \
    --hf-path . \
    --mlx-path ./my-adapted-model \
    --q-bits 4
```

### 7. Test in Python

```python
from mlx_lm import load, generate

model, tokenizer = load("./my-adapted-model")
prompt = "User: I have rice, eggs, tomatoes left. Suggest low-carb dinner under 500 kcal."
response = generate(model, tokenizer, prompt=prompt, max_tokens=200, temp=0.75)
print(response)
```

---

## Tips to Maximize 16 GB

- Close other apps during training.
- Use `--grad-checkpoint` and lower batch/seq-length if you hit memory pressure.
- Start with smaller models and datasets to validate the pipeline before scaling up.
- If runs feel consistently slow, upgrading to a 24 GB config is a significant jump in headroom for 7B+ LoRA or longer sequences.
