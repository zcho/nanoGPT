# AGENTS.md

nanoGPT (karpathy). Deprecated as of Nov 2025 in favor of [nanochat](https://github.com/karpathy/nanochat) — treat this as a
reference/teaching codebase, not something to modernize. No package layout, no tests, no lint/typecheck/CI config, no
`requirements.txt` (deps are listed only in the README).

## Layout

- `model.py` — `GPTConfig` dataclass + `GPT`. Weight-tied `transformer.wte` / `lm_head`. `GPT.from_pretrained()` pulls HF GPT-2
  weights (transposes OpenAI `Conv1D` weights; forces `vocab_size=50257`, `block_size=1024`, `bias=True`; **only** `dropout`
  may be overridden).
- `train.py` — the whole training loop; DDP when `RANK` is set.
- `sample.py`, `bench.py` — generation and a stripped-down benchmark loop.
- `configurator.py` — the config mechanism. `exec`'d into each script's globals; *not* an importable module.
- `config/*.py` — plain assignment files overriding the defaults at the top of `train.py`.
- `data/<dataset>/prepare.py` → `train.bin` / `val.bin` (raw `uint16` memmaps) + optional `meta.pkl` (`itos`, `stoi`, `vocab_size`).
- `scaling_laws.ipynb`, `transformer_sizing.ipynb` — Karpathy's notebooks; unrelated to the training pipeline.

## Commands

- Deps: `pip install torch numpy transformers datasets tiktoken wandb tqdm`. **Ignore the README's `<3` pins** — only
  `datasets<3` is real (`data/openwebtext/prepare.py` uses script-based `load_dataset`, removed in datasets 3). `transformers<3`
  resolves to 2.x and makes pip backtrack through ancient `tokenizers` sdists that fail to build on py3.12 (`distutils` gone).
  Working set: torch 2.14, transformers 4.57, datasets 2.21, numpy 2.5, pyarrow 25. `requests` is needed by both shakespeare
  `prepare.py` scripts but is not listed in the README.
- **Always run from the repo root.** `train.py`/`sample.py`/`bench.py` open `configurator.py`, `data/<dataset>/` and `out_dir`
  through relative paths.
- Fast smoke path (~1MB download, seconds to prepare):
  ```sh
  python data/shakespeare_char/prepare.py
  python train.py config/train_shakespeare_char.py --max_iters=20 --lr_decay_iters=20 --eval_iters=20 --eval_interval=10 --log_interval=1
  python sample.py --out_dir=out-shakespeare-char
  ```
- Real GPT-2 repro (needs 8×A100, ~4 days): `torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py`.
- Finetune: `python train.py config/finetune_shakespeare.py` (defaults to `init_from='gpt2-xl'`).
- `*.bin`, `*.pt`, `*.pkl`, `input.txt` are gitignored, so a fresh clone has **no data**. `python train.py config/eval_gpt2.py`
  still needs `data/openwebtext/{train,val}.bin` (~17GB output, ~54GB HF cache) because `eval_only` runs `estimate_loss()` on both
  splits before exiting.

## Local environment

- `./venv` exists (python 3.12) and has the deps installed, including **CPU-only torch** (`2.14.1+cpu`, installed from
  `https://download.pytorch.org/whl/cpu`) — use `venv/bin/python`. There is no CUDA-enabled torch here.
- The box has 7.5GB RAM (only ~3GB free, swap nearly full) and a 2GB NVIDIA MX450. The MX450 is sm_61, which current CUDA
  builds don't target, and the shipped GPT-2 configs assume an 8×A100 40GB node (`batch_size=12, block_size=1024`).
- Verified-working CPU run (the stock `train_shakespeare_char.py` sizes — 64×256 batch, 6 layers × 384 — get OOM-killed on this
  box, exit 137 with no visible output because stdout was still buffered):
  ```sh
  PYTHONUNBUFFERED=1 venv/bin/python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
    --batch_size=8 --block_size=64 --n_layer=2 --n_head=2 --n_embd=64 \
    --max_iters=3 --lr_decay_iters=3 --eval_iters=2 --eval_interval=2 --log_interval=1
  PYTHONUNBUFFERED=1 venv/bin/python sample.py --out_dir=out-shakespeare-char --device=cpu --num_samples=1 --max_new_tokens=120
  ```
- On CPU / MPS also pass `--compile=False`. Non-`cuda` devices get `device_type='cpu'` (train.py:109), so autocast is skipped
  and `GradScaler` self-disables — the `FutureWarning` at train.py:196 plus "GradScaler is enabled, but CUDA is not available.
  Disabling." is expected noise, not an error.

## configurator.py quirks (easy to get wrong)

- Flags are `--key=value`, not argparse. Space-separated (`--batch_size 12`) hits an `assert` and crashes.
- Values are `literal_eval`'d and **type-checked against the current value**: use `--always_save_checkpoint=False`, not `false`.
  An unknown key raises `ValueError: Unknown config key`.
- Overrides apply left to right: put the config file first (`python train.py config/x.py --batch_size=32`).
- To add a knob, it must exist as a module-level variable in the script's default block (train.py:35-74) before it can be
  overridden; only names captured in `config_keys` (train.py:76) land in the logged/saved `config` dict.

## Training-loop gotchas

- `gradient_accumulation_steps` must be divisible by the DDP world size (assert at train.py:94); it is divided by world size.
- Effective batch = `batch_size × gradient_accumulation_steps × world_size × block_size`. Changing one factor silently changes
  tokens/iter.
- `lr_decay_iters` is separate from `max_iters` — lowering `max_iters` alone means the cosine schedule never completes.
  Keep `lr_decay_iters ≈ max_iters`.
- `always_save_checkpoint = True` (train.py default) overwrites `out_dir/ckpt.pt` at every eval; the shakespeare configs set it
  `False` so only val improvements are kept. Resume with `--init_from=resume --out_dir=<dir>` (model-shape keys are forced
  from the checkpoint; `dropout` and friends stay CLI-controlled).
- Shrinking `--block_size` below the loaded model's uses `crop_block_size` surgery, so finetuning GPT-2 at a shorter context
  still loads pretrained weights.
- Defaults differ between layers: `train.py` uses `bias=False`, while `GPTConfig` defaults `bias=True`. `dtype` defaults to
  bfloat16 when supported, else float16 (which auto-engages `GradScaler`).

## Verifying changes

No test suite exists. Sanity-check edits with: a tiny `GPT` forward + `generate` from `model.py`; the short `train.py` run above
(all three were run and pass); and `python bench.py --real_data=False --compile=False`, which skips data loading but still
requires CUDA — its timing path calls `torch.cuda.synchronize()`, so it raises `AssertionError: Torch not compiled with CUDA
enabled` here. Model dims are hardcoded at bench.py:53 and not configurable.

Match existing style: plain Python, no type hints, single-file modules, `# ---` section separators, `print`-based logging.