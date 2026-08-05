# Trained checkpoint results

Training date: 2026-08-05. Environment: NVIDIA GeForce RTX 5070 12GB, PyTorch 2.11.0, and CUDA 12.8.

Two checkpoints have been generated in the current workspace. The `checkpoints/` directory is ignored by Git to avoid accidentally committing binary files.

| Checkpoint | Size | Best epoch | Training validation-subset exact match |
|---|---:|---:|---:|
| `checkpoints/yt_to_human.pt` | 2.7 MB | 37 | 99.6% |
| `checkpoints/human_to_yt.pt` | 2.7 MB | 40 | 100.0% |

SHA-256 hashes:

```text
81db3db7977b61ae8af014d210a6c5b4e47e3ef579f6568b706b36eb7a2fcc0e  checkpoints/yt_to_human.pt
6d1e3d2d762fdf4b3d46088f19c007bee1459fbe563b7ae279be8f3fc1045cd4  checkpoints/human_to_yt.pt
```

## Training configuration

- 8,000 unique synthetic tableaux with a 0.8/0.1/0.1 train/validation/test split.
- Both row and column human-readable targets.
- At most 5 rows, 8 columns, and 20 cells, with values in `1..50`.
- 671,167 parameters per model.
- `d_model=128`, 8 attention heads, 2 encoder layers, 2 decoder layers, and FFN dimension 256.
- Shared source, target, and output token weights with dropout 0.1.
- Batch size 256, AdamW, learning rate `8e-4`, and at most 40 epochs.
- Seed 42 and training device `cuda:0`.

Reproduction command:

```bash
yt-train \
  --direction both \
  --output-dir checkpoints \
  --device cuda:0 \
  --num-tableaux 8000 \
  --split-ratios 0.8 0.1 0.1 \
  --human-kinds row col \
  --max-rows 5 \
  --max-columns 8 \
  --max-cells 20 \
  --epochs 40 \
  --batch-size 256 \
  --learning-rate 0.0008 \
  --weight-decay 0.0001 \
  --patience 10 \
  --val-exact-limit 256 \
  --d-model 128 \
  --nhead 8 \
  --num-layers 2 \
  --dim-feedforward 256 \
  --dropout 0.1 \
  --max-seq-len 128 \
  --tie-embeddings \
  --seed 42
```

## Full held-out evaluation

Each direction below was evaluated on all 1,600 held-out examples, rather than the 256-example autoregressive validation subset used in training logs.

| Direction | Exact match | Semantic accuracy | Token accuracy | Invalid output |
|---|---:|---:|---:|---:|
| YT to human | 99.5625% | 99.5625% | 99.9845% | 0.1250% |
| Human to YT | 99.4375% | 99.4375% | 99.9814% | 0.0625% |

On 100 unique tableaux, covering 200 row/column `raw -> human -> raw` round trips:

- Exact match: 99.5%.
- Invalid pipeline rate: 0%.

## Screenshot example

Actual model output:

```text
input:  [YT start] 2 3 5 | 1 4 [YT end]
row:    [YT row start] 2 3 5 | 1 4 [YT row end]
col:    [YT col start] 2 1 | 3 4 | 5 [YT col end]
back:   [YT start] 2 3 5 | 1 4 [YT end]
```
