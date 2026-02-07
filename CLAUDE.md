# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a minimal, hackable experimental harness for training LLMs from scratch. It trains compute-optimal models using a single complexity dial (`--depth`) that automatically determines all other hyperparameters via principled scaling laws. The goal: train GPT-2 capability models (~1.6B params) in ~3 hours on 8xH100 for <$100.

## Essential Commands

### Training Commands

**Quick experimentation (5 min, d12 model on 8 GPUs):**
```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

**GPT-2 speedrun (3 hours, d26 model):**
```bash
bash runs/speedrun.sh
```

**Single GPU training (8x slower):**
```bash
python -m scripts.base_train --depth=12 --run="d12" --model-tag="d12"
```

### Evaluation Commands

**Evaluate base model:**
```bash
python -m scripts.base_eval \
    --checkpoint-path=base_checkpoints/d26/model_007226.pt \
    --eval=core,bpb,sample
```

**Chat with trained model (web UI):**
```bash
python -m scripts.chat_web
# Then visit http://localhost:8000 or http://<public-ip>:8000
```

**Chat CLI:**
```bash
python -m scripts.chat_cli --checkpoint-path=chatsft_checkpoints/d26/model_000200.pt
```

### Testing

```bash
pytest tests/
pytest tests/test_engine.py  # Single test file
pytest -m "not slow"  # Skip slow tests
```

## Core Architecture

### The Single Complexity Dial: `--depth`

The defining feature of nanochat is that users specify ONLY `--depth` (number of transformer layers). All other hyperparameters are derived automatically:

1. **Model width**: `model_dim = depth × aspect_ratio` (nudged to be divisible by head_dim)
2. **Training duration**: Chinchilla-style `target_tokens = target_param_data_ratio × num_params`
3. **Batch size**: Power Lines paper scaling `Bopt ∝ D^0.383`, auto-computed as nearest power-of-2
4. **Learning rate**: `lr = lr_ref × √(B/B_ref)` scaling from reference batch size
5. **Weight decay**: T_epoch framework `λ = λ_ref × √(B/B_ref) × (D_ref/D)` for consistency across depths

Reference model is **d12** (GPT-1 sized) with base hyperparameters that transfer to all depths via muP-style scaling.

### Training Pipeline: Base → SFT → RL

```
Pretraining (base_train.py)          SFT (chat_sft.py)              RL (chat_rl.py)
300B tokens diverse data    →    Task mixture + SmolTalk    →    Optional refinement
BOS-bestfit packing                BOS-bestfit with padding
Saves to base_checkpoints/         Saves to chatsft_checkpoints/
```

**Key files:**
- `scripts/base_train.py` - Pretraining orchestration, hyperparameter calculation, training loop
- `scripts/chat_sft.py` - Supervised fine-tuning on task mixtures
- `nanochat/gpt.py` - Transformer model (FlashAttention3, RoPE, GQA, value embeddings, sliding window)
- `nanochat/optim.py` - MuonAdamW optimizer (Muon for matrices, AdamW for embeddings/scalars)
- `nanochat/dataloader.py` - BOS-aligned best-fit packing dataloader (~100% token utilization)

### Model Architecture (gpt.py)

Modern GPT with optimizations:
- **No positional embeddings** (RoPE only)
- **QK normalization** in attention
- **Untied embeddings** (separate token embedding and lm_head)
- **ReLU² activation** in MLP
- **Group-Query Attention** for efficient inference (configurable kv_heads)
- **Sliding window attention** with pattern-based config (e.g., "SSSL" = local, local, local, global)
- **Value embeddings** (ResFormer-style, alternating layers)
- **Per-layer learnable scalars**: `resid_lambdas` (residual scaling), `x0_lambdas` (input embedding blending)
- **FlashAttention 3** with KV caching (falls back to FA2/SDPA)

### Hyperparameter Transfer System

Located in `base_train.py:get_depth_adjusted_hparams()`:

1. **Model sizing**: Uses `aspect_ratio` to compute `model_dim` from depth
2. **Compute-optimal horizon**: `target_tokens = target_param_data_ratio × scaling_params`
3. **Batch size scaling**: `B = round_to_power_of_2(B_ref × (D/D_ref)^0.383)` from Power Lines paper
4. **LR scaling**: `lr = lr_ref × √(B/B_ref)` to maintain optimization trajectory
5. **Weight decay scaling**: T_epoch framework maintains consistent regularization landscape
6. **Muon LR scaling**: Different coefficient (`muon_factor = 0.38`) due to momentum differences

All depths (d6 to d52+) work out-of-the-box with compute-optimal training.

### Data Pipeline: BOS-Bestfit Packing

**Algorithm** (in `dataloader.py`):
1. Every sequence starts with BOS token
2. Greedily pack documents using best-fit: find largest document ≤ remaining space
3. When no document fits, crop/pad remaining space
4. Result: ~100% token utilization, ~35% token cropping for pretraining

**For SFT**: Uses padding instead of cropping to preserve all tokens in conversations.

**Distributed**: DDP-aware with per-rank sharding and deterministic shuffling. Resume support via `(pq_idx, rg_idx, epoch)` position tracking.

### Evaluation System

**Two main metrics:**

1. **val_bpb** (bits per byte): Loss normalized by token byte length, vocab-size-agnostic
   - Computed in `loss_eval.py:evaluate_bpb()`
   - Ignores special tokens and masked positions (ignore_index=-1)

2. **CORE metric** (DCLM benchmark): Ensemble score over 22 tasks (ARC, MMLU, etc.)
   - Computed in `core_eval.py`
   - Multiple modes: multiple-choice, schema questions, language modeling
   - Target: Beat GPT-2's 0.256525 score

**Task system** (`tasks/`):
- Base `Task` class with slicing support
- `TaskMixture` combines and shuffles tasks deterministically
- Implementations: MMLU, GSM8K, SmolTalk, ARC, HumanEval, SpellingBee, CustomJSON
- Each provides: `eval_type`, `get_example()`, `evaluate()` methods

### Inference Pipeline

**Engine** (`engine.py`):
- Batch=1 prefill → replicate KV cache for num_samples → batched decode
- FA3-compatible KV cache (B, T, H, D) layout
- Sampling: temperature, top-k, greedy
- Tool use state machine for Python code execution with safety checks

**Web UI** (`chat_web.py`):
- FastAPI server with worker pool for multi-GPU data parallelism
- Queue-based request distribution to available workers
- Streaming responses via SSE
- Endpoints: `/` (UI), `/chat/completions`, `/health`, `/stats`
- Abuse prevention: limits on message count, length, conversation length

### Checkpoint Format

**Structure:**
- `model_{step:06d}.pt` - Model state_dict
- `optim_{step:06d}_rank{rank}.pt` - Per-rank optimizer state (sharded)
- `meta_{step:06d}.json` - Metadata with model_config, training state, dataloader position

**Compatibility:** Handles missing keys from old checkpoints, patches new features, converts bf16→fp32 for CPU inference.

### FP8 Training

**Flag:** `--fp8` (requires GPU with fp8 support, e.g., H100)

Converts Linear layers to Float8Linear via torchao (dims must be divisible by 16). Lower precision per step but faster iteration. Falls back to bf16 during evaluation. Models trained in fp8 vs bf16 have slightly different quality/speed tradeoffs.

### Distributed Training

**DDP via torchrun:**
```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- <args>
```

- Rank-aware data loading with automatic sharding
- Gradient accumulation computed automatically if device_batch_size < target per-GPU batch
- Optimizer state sharded across ranks
- Synchronized training state for fault tolerance

## Development Patterns

### Key Hyperparameters

- `--depth`: Number of transformer layers (THE dial: 6-52+)
- `--device-batch-size`: Per-GPU batch size (default 32, reduce to 16/8/4 if OOM)
- `--total-batch-size`: Override auto-computed batch size (in tokens)
- `--target-param-data-ratio`: Tokens:Params ratio (default 10.5 for compute-optimal)
- `--run`: wandb run name
- `--model-tag`: Checkpoint directory name
- `--fp8`: Enable fp8 training (H100 only)

### Monitoring Training (wandb)

Key metrics to watch:
1. `val_bpb` vs `step`, `total_training_time`, `total_training_flops`
2. `core_metric` (target: >0.256525 for GPT-2)
3. `train/mfu` (Model FLOPs Utilization), `train/tok_per_sec` (throughput)
4. VRAM utilization

### Common Adjustments

**Reduce memory usage:** Lower `--device-batch-size` (32→16→8→4). Script auto-compensates with gradient accumulation.

**Undertrain/overtrain a model:** Adjust `--target-param-data-ratio` (default 10.5). Lower = undertrain, higher = overtrain.

**Change batch size:** Use `--total-batch-size` (in tokens). Must be power-of-2 for clean gradient accumulation.

**Disable periodic evals:** `--sample-every=-1 --save-every=-1 --core-metric-every=999999`

### Leaderboard Runs

To submit a speedrun result:
1. Train d24-d26 model with target CORE > 0.256525
2. Report `total_training_time` (from wandb, excludes eval/logging overhead)
3. Report `val_bpb` and `core_metric`
4. Changes must generalize to all depths (entire miniseries), not just target a single model
5. See `dev/LEADERBOARD.md` for detailed submission process

### Scaling Laws Scripts

- `runs/miniseries.sh` - Train entire compute-optimal model series (d6, d8, d10, ..., d32)
- `runs/scaling_laws.sh` - Experiment with different hyperparameters across depths
- See discussions: [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)

## Important Design Principles

1. **Simplicity over configurability**: Not an exhaustive LLM framework, but a minimal hackable baseline
2. **Single dial philosophy**: Users shouldn't tune hyperparameters, only specify depth
3. **Principled scaling**: Changes must work across all depths via scaling laws, not just one model
4. **Compute efficiency**: Optimize wall-clock time, not just loss curves
5. **End-to-end completeness**: Tokenization → pretraining → SFT → RL → inference → chat UI

## Code Organization

- `nanochat/` - Core library (model, dataloader, optimizer, eval, inference engine)
- `scripts/` - Entry points (base_train, chat_sft, chat_web, eval scripts)
- `tasks/` - Task definitions for evaluation and SFT
- `runs/` - Shell scripts for common workflows (speedrun, miniseries, scaling laws)
- `dev/` - Development utilities (synthetic data generation, data preprocessing)
- `tests/` - pytest test suite
