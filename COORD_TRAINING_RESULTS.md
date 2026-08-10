# Coordinate-notation training results

Training date: 2026-08-10. Environment: NVIDIA GeForce RTX 5070 12 GB,
PyTorch 2.11.0+cu128, and CUDA 12.8.

Two coordinate-only checkpoints were trained and saved separately from the
earlier row/column models:

| Checkpoint | Size | Best epoch | Validation-subset exact match |
|---|---:|---:|---:|
| `checkpoints/coord/yt_to_human.pt` | 2.8 MB | 32 | 100% |
| `checkpoints/coord/human_to_yt.pt` | 2.8 MB | 40 | 100% |

SHA-256 hashes:

```text
33f1c4cc61c1b2084299450ce5f9db80b8328fdf88d0e735b366372a48871453  checkpoints/coord/yt_to_human.pt
ed5c28701b90e2b45c046c5d61ce1148a8bb559655dc39ca7313dde1b7c3a7cb  checkpoints/coord/human_to_yt.pt
```

## Coordinate convention

Coordinates are 1-based `(row,column)` matrix coordinates and entries are
listed in row-major order. Surface text uses decimal integers; `n1..n50` are
the corresponding internal tokenizer symbols. For example:

```text
raw:   [YT start] 2 3 5 | 1 4 [YT end]
coord: [YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | (2,1) : 1 | (2,2) : 4 [YT coord end]
```

## Training configuration

- 8,000 unique synthetic tableaux with a `0.8/0.1/0.1`
  train/validation/test split.
- Coordinate notation only, producing 6,400/800/800 examples per direction.
- At most 5 rows, 8 columns, and 20 cells, with values in `1..50`.
- 672,070 parameters per model and a 70-token vocabulary.
- `d_model=128`, 8 attention heads, 2 encoder layers, 2 decoder layers, and
  FFN dimension 256.
- Shared source, target, and output token weights with dropout 0.1.
- Maximum sequence length 256; a 20-cell coordinate target uses 243 tokens.
- Batch size 128, AdamW, learning rate `8e-4`, and 40 epochs.
- Seed 42 and training device `cuda:0`.

Reproduction command:

```bash
yt-train \
  --direction both \
  --output-dir checkpoints/coord \
  --device cuda:0 \
  --num-tableaux 8000 \
  --split-ratios 0.8 0.1 0.1 \
  --human-kinds coord \
  --max-rows 5 \
  --max-columns 8 \
  --max-cells 20 \
  --epochs 40 \
  --batch-size 128 \
  --learning-rate 0.0008 \
  --weight-decay 0.0001 \
  --grad-clip 1.0 \
  --patience 10 \
  --val-exact-limit 256 \
  --d-model 128 \
  --nhead 8 \
  --num-layers 2 \
  --dim-feedforward 256 \
  --dropout 0.1 \
  --max-seq-len 256 \
  --tie-embeddings \
  --seed 42
```

## Full held-out evaluation

Each direction was greedily decoded on all 800 unseen examples.

| Direction | Exact match | Semantic accuracy | Token accuracy | Invalid output |
|---|---:|---:|---:|---:|
| YT to coordinate | 100% | 100% | 100% | 0% |
| Coordinate to YT | 100% | 100% | 100% | 0% |

Teacher-forced held-out losses were `0.000222466` for YT to coordinate and
`0.000263793` for coordinate to YT. On 100 unique tableaux, all 100
`raw -> coord -> raw` round trips were exact and none produced an invalid
intermediate output.

Evaluation command:

```bash
yt-evaluate \
  --checkpoint checkpoints/coord/yt_to_human.pt \
  --checkpoint checkpoints/coord/human_to_yt.pt \
  --device cuda:0 \
  --batch-size 128 \
  --round-trip-limit 100
```

## Verified example

Actual model output:

```text
input: [YT start] 2 3 5 | 1 4 [YT end]
coord: [YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | (2,1) : 1 | (2,2) : 4 [YT coord end]
back:  [YT start] 2 3 5 | 1 4 [YT end]
```

The checkpoints are ignored by Git, like the earlier binary model files, so
the hashes above identify the local trained artifacts.
