# Human-readable notation for YT

This project turns the **YT (Young tableau) notation-conversion task** from the original screenshot into a complete and reproducible small-scale experiment. It uses a fixed hand-written tokenizer and synthetic data to train two encoder-decoder Transformers:

- `yt_to_human`: converts raw YT text into row- or column-based human-readable notation.
- `human_to_yt`: converts row- or column-based human-readable notation back into raw YT text.

The project also provides a deterministic converter that does not use a model. The deterministic converter serves as the label-generation and correctness oracle, while the Transformers support the experimental goal of training small models to learn the conversion.

Two trained checkpoints are available in the current workspace. See [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md) for their training configuration, hashes, and complete held-out metrics.

Permutation OOD experiments show that the models do not generalize to the combined distribution of longer inputs with all-unique values. Because those experiments change length, shape, and filling simultaneously, see [`OOD_RESULTS.md`](OOD_RESULTS.md) for the full results and proposed controlled experiments.

## Scope and explicit assumptions

- `YT` means Young tableau in this repository, not YouTube.
- Model inputs are **canonical plain text**, not images, screenshots, handwritten tables, or LaTeX.
- A tableau is internally represented as rows ordered from top to bottom, with each row read from left to right and all rows left-aligned.
- Row lengths must be non-increasing. For example, `(3, 2, 1)` is valid, while `(2, 3)` is invalid.
- Each cell contains an integer in `1..50`. The current task does not require unique values or enforce row/column monotonicity, so the generator does not produce standard or semistandard Young tableaux.
- `row` reads one row at a time. `col` reads columns from left to right and each column from top to bottom, skipping missing cells in ragged shapes.
- Surface formats are strictly canonical: they use ASCII spaces, exactly one space between adjacent surface tokens, and ` | ` as the group separator.
- Synthetic data is deduplicated and split by the underlying tableau before row/column forms are expanded. Different textual representations of the same tableau therefore cannot leak across train, validation, and test splits.

## Three text formats

The same tableau from the screenshot is:

```text
2 3 5
1 4
```

The repository supports three canonical surface forms:

```text
raw: [YT start] 2 3 5 | 1 4 [YT end]
row: [YT row start] 2 3 5 | 1 4 [YT row end]
col: [YT col start] 2 1 | 3 4 | 5 [YT col end]
```

The bodies of `raw` and `row` are both row-major, but they use different markers. The three groups in `col` represent columns `(2, 1)`, `(3, 4)`, and `(5)`. All three formats can be parsed losslessly into the same tableau.

## Hand-written tokenizer

The vocabulary contains 63 fixed tokens whose IDs are not learned:

| ID | Token | Surface text or purpose |
|---:|---|---|
| 0 | `PAD` | Batch padding |
| 1 | `BOS` | Sequence start |
| 2 | `EOS` | Sequence end |
| 3 | `TO_ROW` | Control token requesting row output |
| 4 | `TO_COL` | Control token requesting column output |
| 5 | `x1` | `[YT row start]` |
| 6 | `x2` | `[YT row end]` |
| 7 | `x3` | `[YT col start]` |
| 8 | `x4` | `[YT col end]` |
| 9 | `x5` | `[YT start]` |
| 10 | `x6` | `[YT end]` |
| 11 | `s` | One ASCII space |
| 12 | `x` | `|` |
| 13..62 | `n1`..`n50` | Integers `1`..`50`; the ID of `nK` is `K + 12` |

For example, row notation is encoded as:

```text
BOS x1 s n2 s n3 s n5 s x s n1 s n4 s x2 EOS
```

The tokenizer strictly parses canonical notation before encoding it. Unknown markers, noncanonical whitespace, leading zeroes, and numbers outside `1..50` are rejected instead of being silently normalized.

### Why forward conversion needs a control token

The same raw input has two valid targets:

```text
raw + TO_ROW -> row notation
raw + TO_COL -> col notation
```

Without a control token, an identical source would map to two different targets and the model would not know which form to generate. During encoding, the control token appears immediately after `BOS`:

```text
BOS TO_ROW x5 ... x6 EOS
BOS TO_COL x5 ... x6 EOS
```

The reverse model does not need an additional control token because the first input marker already disambiguates the format: row notation begins with `x1`, column notation begins with `x3`, and both map to raw notation.

## Installation

Python 3.10 or later and PyTorch 2.1 or later are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you need a specific CUDA build, install the appropriate PyTorch package for your platform first, then run `python -m pip install -e .`. The installation exposes the `yt-convert`, `yt-train`, `yt-infer`, `yt-evaluate`, and `yt-ood` commands.

## Deterministic conversion

Deterministic conversion does not load a checkpoint and cannot introduce model errors. It is useful for validating formats, generating labels, and debugging the tokenizer.

Convert raw notation to column notation:

```bash
yt-convert '[YT start] 2 3 5 | 1 4 [YT end]' --to col
```

Output:

```text
[YT col start] 2 1 | 3 4 | 5 [YT col end]
```

Convert column notation back to raw notation:

```bash
yt-convert '[YT col start] 2 1 | 3 4 | 5 [YT col end]' --to raw
```

Valid values for `--to` are `raw`, `row`, and `col`.

## Training both models

### Full training

The following command uses the main default configuration: 4,000 unique tableaux, a `0.8/0.1/0.1` split, both row and column targets, and at most 15 epochs. `auto` prefers CUDA, then MPS, and finally falls back to CPU.

```bash
yt-train \
  --direction both \
  --output-dir checkpoints \
  --device auto \
  --num-tableaux 4000 \
  --split-ratios 0.8 0.1 0.1 \
  --human-kinds row col \
  --epochs 15 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --patience 5 \
  --val-exact-limit 256 \
  --seed 42
```

Training saves the best validation checkpoint for each direction:

```text
checkpoints/yt_to_human.pt
checkpoints/human_to_yt.pt
```

The default model uses `d_model=64`, 4 attention heads, 2 encoder layers, 2 decoder layers, a feed-forward dimension of 128, dropout 0.1, and a maximum sequence length of 128. Because the input and output use the same fixed vocabulary, source embeddings, target embeddings, and output-projection token weights are shared by default. Use `--no-tie-embeddings` to disable sharing. Each checkpoint stores its direction, model configuration, fixed vocabulary, best epoch, metrics, and synthetic-data configuration so that evaluation can reconstruct the exact held-out split.

To train only one direction, replace `--direction both` with `yt_to_human` or `human_to_yt`.

### Quick smoke training

The following CPU command verifies the end-to-end pipeline. It is not intended to produce a high-quality model:

```bash
yt-train \
  --direction both \
  --output-dir checkpoints/quick \
  --device cpu \
  --num-tableaux 96 \
  --max-rows 3 \
  --max-columns 4 \
  --max-cells 8 \
  --epochs 2 \
  --batch-size 32 \
  --patience 0 \
  --val-exact-limit 24 \
  --d-model 32 \
  --nhead 4 \
  --num-layers 1 \
  --dim-feedforward 64 \
  --dropout 0 \
  --max-seq-len 64 \
  --seed 42
```

`--patience 0` disables early stopping. Training seeds Python and PyTorch, and synthetic data generation and splitting are deterministic. Floating-point results are not guaranteed to be bit-for-bit identical across hardware or PyTorch kernels.

## Inference in both directions

These commands use greedy decoding. A quick smoke checkpoint may not have learned the task. If a model does not produce valid markers or emit `EOS` before the length limit, the command reports a clear error and displays the generated tokens.

Convert raw notation to row notation:

```bash
yt-infer \
  --checkpoint checkpoints/yt_to_human.pt \
  --text '[YT start] 2 3 5 | 1 4 [YT end]' \
  --style row \
  --device auto
```

Convert raw notation to column notation:

```bash
yt-infer \
  --checkpoint checkpoints/yt_to_human.pt \
  --text '[YT start] 2 3 5 | 1 4 [YT end]' \
  --style col \
  --device auto
```

Convert human-readable column notation back to raw notation:

```bash
yt-infer \
  --checkpoint checkpoints/human_to_yt.pt \
  --text '[YT col start] 2 1 | 3 4 | 5 [YT col end]' \
  --device auto
```

For a `human_to_yt` checkpoint, the program determines row versus column input from the marker, so `--style` does not affect reverse conversion. The optional `--max-new-tokens N` argument can reduce the generation limit; by default, inference uses the checkpoint's model sequence limit.

## Evaluating checkpoints

Evaluate one checkpoint:

```bash
yt-evaluate \
  --checkpoint checkpoints/yt_to_human.pt \
  --device auto \
  --batch-size 64
```

Evaluate both checkpoints and measure raw-to-human-to-raw round trips:

```bash
yt-evaluate \
  --checkpoint checkpoints/yt_to_human.pt \
  --checkpoint checkpoints/human_to_yt.pt \
  --device auto \
  --batch-size 64 \
  --round-trip-limit 100
```

Pass `--checkpoint` once or twice. When supplied twice, the checkpoints must represent different directions. JSON output for each direction contains:

- `loss` and teacher-forced `token_accuracy`.
- Greedy-decoding `exact_match`.
- `semantic_accuracy`, which requires both the parsed tableau and target format to be correct.
- `invalid_output_rate` for outputs that cannot be parsed or do not emit `EOS`.

When both directions are available, the report also contains `round_trip.exact_match` and `invalid_pipeline_rate`. `--limit N` limits held-out examples per direction, while `--round-trip-limit N` limits the number of unique tableaux used for round-trip evaluation.

## Running tests

All tests use the standard-library `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

The suite covers canonical formatting and parsing, fixed token IDs, reproducible data generation, split leakage prevention, model shapes and decoding, checkpoint validation, deterministic conversion, and the main training, inference, evaluation, and OOD paths.

## Permutation tests beyond the training length

The trained models only saw tableaux with at most 20 cells. The following command uses 20 cells as a baseline, then tests random all-unique permutations with 21, 30, 40, and 50 cells, plus a 54-cell repeated-value sequence-length stress case:

```bash
yt-ood \
  --yt-to-human checkpoints/yt_to_human.pt \
  --human-to-yt checkpoints/human_to_yt.pt \
  --entries 20 21 30 40 50 54 \
  --samples 100 \
  --device cuda:0
```

Because the hand-written vocabulary is strictly limited to `n1..n50`, 50 is the maximum permutation length with all distinct entry values. The 54-cell case repeats some `1..50` tokens and only tests sequence length. Supporting more than 50 distinct values requires a compositional digit-level tokenizer and retraining.

## Project layout

| Path | Purpose |
|---|---|
| `yt_transformer/notation.py` | `Tableau` and strict parsing/serialization of all three formats |
| `yt_transformer/tokenizer.py` | The 63-token hand-written vocabulary and encode/decode logic |
| `yt_transformer/data.py` | Reproducible synthetic tableaux, paired examples, splits, Dataset, and collation |
| `yt_transformer/model.py` | Small batch-first encoder-decoder Transformer |
| `yt_transformer/train.py` | Training, validation, and early stopping for both directions |
| `yt_transformer/checkpoint.py` | Atomic, versioned checkpoint saving and loading |
| `yt_transformer/convert.py` | Deterministic model-free format conversion |
| `yt_transformer/infer.py` | Single-example checkpoint inference |
| `yt_transformer/evaluate.py` | Held-out metrics and two-model round-trip evaluation |
| `yt_transformer/ood.py` | Permutation and length-extrapolation tests beyond training cell counts |
| `yt_transformer/runtime.py` | Random seeds and CPU/CUDA/MPS device selection |
| `tests/` | Standard-library unit test suite |

## Current limitations

- The project only processes canonical **plain text**. It cannot directly read grid images, chat screenshots, OCR output, or LaTeX. Such inputs require a separate vision/OCR/parser stage that produces raw notation.
- This repository implements only the first task from the screenshot. **Task 2, large-number representation and `A + B = C` addition**, is not implemented, and no base-50 or bijective-base-50 arithmetic model has been trained.
- The vocabulary only covers integers `1..50`; models should only be used within the shape and sequence-length limits configured during training.
- Synthetic fillings do not satisfy the uniqueness or monotonicity rules of standard or semistandard Young tableaux. If those mathematical constraints are required, replace the data-generation rules and retrain.
- Neural models cannot guarantee valid or correct output. Use `yt-convert` when conversion must be exact, and use held-out plus round-trip evaluation when measuring learned-model behavior.
- Checkpoints are loaded with PyTorch's `weights_only=True` and validated for version, vocabulary, and configuration, but only checkpoints created locally or obtained from a trusted source should be loaded. Do not treat arbitrary uploaded files as safe inputs.
