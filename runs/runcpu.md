# Here's what runs/runcpu.sh does — end to end:             

  ---                                                                                                                                           
##  What the script does
                                                                                                                                                
###  Stage 1 — Environment setup                               

  - Installs uv if missing
  - Checks for cargo (Rust toolchain) — exits with clear instructions if absent
  - Creates a .venv and runs uv sync --extra cpu, which compiles rustbpe from source (~20-25s first time, cached after)

###  Stage 2 — Data download

  python -m nanochat.dataset -n 8
  Downloads 8 parquet shards from HuggingFace (karpathy/climbmix-400b-shuffle) into ~/.cache/nanochat/data/. This is a diverse web-text corpus —
   news, code, multilingual content — used for base model pretraining.

###  Stage 3 — Tokenizer training

  python -m scripts.tok_train --max-chars=2000000000
  python -m scripts.tok_eval
  - Trains a BPE tokenizer on 2 billion characters using our rustbpe — Rayon parallelism uses all CPU cores (~25-35s on M-series)
  - Saves tokenizer.pkl + token_bytes.pt to ~/.cache/nanochat/tokenizer/
  - tok_eval benchmarks compression ratio vs GPT-2 and GPT-4 across 5 domains (news, Korean, code, math, science) — gives you a quality report on the tokenizer

###  Stage 4 — Base model training (~30 min on M3 Max)

  python -m scripts.base_train \
      --depth=6 --head-dim=64 --window-pattern=L \
      --max-seq-len=512 --num-iterations=5000 ...
  Trains a 6-layer GPT-style transformer from scratch using the tokenizer just trained. Key parameters:
  - Sliding-window attention (--window-pattern=L) for memory efficiency
  - 512-token context, batch 32, total batch 16K tokens
  - Evaluates every 100 steps, samples generated text every 100 steps
  - Saves a checkpoint to runs/

  python -m scripts.base_eval
  Evaluates the trained base model on held-out data.

###  Stage 5 — Supervised Fine-Tuning (~10 min on M3 Max)

  curl ... identity_conversations.jsonl
  python -m scripts.chat_sft ...
  Downloads ~identity Q&A conversation data and fine-tunes the base model for chat/instruction following — this is what turns "next token predictor" into "assistant that answers questions."

###  Stage 6 — Chat interface (commented out, run manually)

  python -m scripts.chat_cli -p "What is the capital of France?"
  python -m scripts.chat_web
  - CLI: single-turn command-line chat
  - Web: ChatGPT-style browser UI

  ---
  What you get at the end

  A fully self-trained small language model — tokenizer, base weights, and SFT weights — built entirely from scratch on your local machine. It's educational quality (not GPT-4 quality), but it demonstrates the complete pretraining → SFT pipeline that powers production LLMs. The model will handle simple factual Q&A and basic conversation.

  What makes it meaningful for our work

  The tokenizer stage now uses our local rustbpe with all 6 fixes — deterministic training, correct i64 pair counts, O(n log n) encode, and lean memory layout — rather than the PyPI version with known bugs. The rest of the pipeline (tok_train.py, tokenizer.py) calls rustbpe.Tokenizer() directly, so it automatically gets the fixed implementation without any Python changes.