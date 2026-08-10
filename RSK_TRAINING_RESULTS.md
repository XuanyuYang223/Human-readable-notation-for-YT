# RSK permutation-to-YT training results

This run trains a separate encoder-decoder Transformer to map a canonical
permutation to the standard Robinson--Schensted insertion tableau `P`:

```text
[perm start] 3 5 1 4 2 [perm end]
-> [YT start] 1 2 | 3 4 | 5 [YT end]
```

Only `P` is predicted. There is no reverse task because recovering the source
permutation also requires the recording tableau `Q`.

## Run status

Training was stopped at the user's request while epoch 11 was in progress. The
atomic best-checkpoint file therefore remains the completed epoch-10 model.
It is a useful partial result rather than a claim that the planned 50-epoch run
converged.

The executed configuration was:

```bash
python -m yt_transformer.rsk_train \
  --output-dir checkpoints/rsk \
  --device cuda:0 \
  --num-permutations 80000 \
  --min-length 1 \
  --max-length 20 \
  --split-ratios 0.8 0.1 0.1 \
  --split-seed 43 \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --patience 10 \
  --val-exact-limit 512 \
  --d-model 256 \
  --nhead 8 \
  --num-layers 4 \
  --dim-feedforward 1024 \
  --dropout 0.1 \
  --max-seq-len 128 \
  --tie-embeddings \
  --seed 42
```

Hardware/software for this run:

- NVIDIA GeForce RTX 5070, 12,227 MiB reported VRAM
- PyTorch 2.11.0+cu128
- 7,392,328 unique model parameters
- 72-token append-only RSK vocabulary
- about 26.7 seconds per completed epoch

## Data and leakage control

The generator sampled 80,000 unique permutations with length-balanced quotas
over lengths 1 through 20. Identity and reverse permutations are included at
every represented length. Labels come from the deterministic row-insertion
oracle, not from another learned model.

Splits contain 63,997 train, 8,003 validation, and 8,000 test examples. All
permutations with the same insertion tableau `P` are assigned together, so the
same target/Knuth class cannot cross split boundaries. Assignment is stratified
by permutation length. Lengths 3 through 20 occur in all three splits; lengths
1 and 2 cannot populate three `P`-disjoint splits and are train-only.

## Saved validation result

The checkpoint records the balanced, length-interleaved 512-example validation
prefix used for autoregressive model selection:

| Metric | Epoch 10 |
|---|---:|
| Validation loss | 0.0161666 |
| Teacher-forced token accuracy | 99.3813% |
| Greedy exact match | 81.6406% |

## Full held-out result

The saved epoch-10 checkpoint was reloaded and evaluated by greedy decoding on
all 8,000 `P`-group-disjoint test examples:

```bash
python -m yt_transformer.rsk_evaluate \
  --checkpoint checkpoints/rsk/perm_to_yt.pt \
  --device cuda:0 \
  --batch-size 128
```

| Metric | Full test |
|---|---:|
| Examples | 8,000 |
| Loss | 0.0160201 |
| Teacher-forced token accuracy | 99.3826% |
| Greedy surface exact match | 81.1500% |
| Tableau semantic accuracy | 81.1500% |
| Shape exact match | 81.6750% |
| Value-content preservation | 99.1375% |
| Invalid output rate | 0.6875% |

Exact accuracy is strongly length-dependent, which is expected for this
interrupted run:

| Length | Examples | Exact | Shape exact | Content preserved | Invalid | Token accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 0.00% | 0.00% | 0.00% | 0.00% | 92.86% |
| 4 | 2 | 50.00% | 50.00% | 50.00% | 0.00% | 96.43% |
| 5 | 12 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| 6 | 72 | 98.61% | 98.61% | 98.61% | 1.39% | 99.93% |
| 7 | 503 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| 8 | 570 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| 9 | 570 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| 10 | 570 | 99.12% | 99.12% | 100.00% | 0.00% | 99.97% |
| 11 | 570 | 99.30% | 99.30% | 100.00% | 0.00% | 99.98% |
| 12 | 570 | 96.67% | 96.67% | 99.65% | 0.35% | 99.91% |
| 13 | 570 | 94.56% | 94.56% | 99.65% | 0.35% | 99.85% |
| 14 | 570 | 91.75% | 91.75% | 99.82% | 0.18% | 99.79% |
| 15 | 570 | 79.82% | 80.18% | 100.00% | 0.00% | 99.50% |
| 16 | 570 | 73.68% | 73.86% | 99.12% | 0.88% | 99.35% |
| 17 | 570 | 64.39% | 64.56% | 97.54% | 1.58% | 99.11% |
| 18 | 570 | 56.49% | 57.19% | 98.42% | 0.70% | 98.94% |
| 19 | 570 | 47.54% | 50.53% | 97.19% | 2.63% | 98.67% |
| 20 | 570 | 32.63% | 35.61% | 97.02% | 2.81% | 98.14% |

The tiny length-3/4 buckets contain only one and two examples because a strict
`P`-group split has very few mathematical groups at those lengths; their
percentages should not be interpreted as stable estimates.

## Artifact integrity and availability

The local checkpoint is:

```text
path: checkpoints/rsk/perm_to_yt.pt
size: 29,754,060 bytes
SHA-256: c43ae5c0ad78682b4e61931c404fdf27e5fd1378d017eedcc1a71983e5f480eb
checkpoint epoch: 10
```

`checkpoints/` is intentionally ignored by Git, so the repository commit
contains the reproducible code, configuration, metrics, and hash but not this
29.8 MB binary. The hash identifies the current local artifact; retraining is
seeded but is not promised to be bit-for-bit identical across GPU/PyTorch
kernels.

## Interpretation

This partial run already learns the task well through length 14, but it has not
converged at lengths 15 through 20. Teacher-forced token accuracy is much higher
than whole-sequence exact match because a single wrong token makes the entire
tableau prediction non-exact. Continue training or add broader shape-balanced
coverage before treating length-20 performance as production quality. For
guaranteed-correct conversion, use the deterministic `yt-rsk` command.
