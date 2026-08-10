# Human-readable notation for YT

This project turns the **YT (Young tableau) tasks** from the original prompt into reproducible small-scale experiments. It uses fixed hand-written tokenizers and synthetic data to train encoder-decoder Transformers:

- `yt_to_human`: converts raw YT text into row-, column-, or coordinate-based human-readable notation.
- `human_to_yt`: converts row, column, or coordinate notation back into raw YT text.
- `perm_to_yt`: learns the Robinson--Schensted map from a permutation to its insertion tableau `P`.

The project also provides deterministic converters that do not use a model. They serve as label-generation and correctness oracles, while the Transformers support the experimental goal of learning each conversion.

[`TRAINING_RESULTS.md`](TRAINING_RESULTS.md) records the earlier row/column-only
training run, while [`COORD_TRAINING_RESULTS.md`](COORD_TRAINING_RESULTS.md)
records the new coordinate-only run and its 100% held-out exact-match results.
The original 63-token checkpoints remain loadable for row/column inference;
they cannot generate coordinate notation, which requires the new 70-token
checkpoints under `checkpoints/coord/`.
[`RSK_TRAINING_RESULTS.md`](RSK_TRAINING_RESULTS.md) records the separate
permutation-to-tableau run. Its checkpoint uses a 72-token vocabulary and lives
under `checkpoints/rsk/`; it is not interchangeable with the notation models.

Permutation OOD experiments on the earlier row/column models show that they do
not generalize to the combined distribution of longer inputs with all-unique
values. Because those experiments change length, shape, and filling
simultaneously, see [`OOD_RESULTS.md`](OOD_RESULTS.md) for the full results and
proposed controlled experiments.

## Scope and explicit assumptions

- `YT` means Young tableau in this repository, not YouTube.
- Model inputs are **canonical plain text**, not images, screenshots, handwritten tables, or LaTeX.
- A tableau is internally represented as rows ordered from top to bottom, with each row read from left to right and all rows left-aligned.
- Row lengths must be non-increasing. For example, `(3, 2, 1)` is valid, while `(2, 3)` is invalid.
- Each cell contains an integer in `1..50`. The current task does not require unique values or enforce row/column monotonicity, so the generator does not produce standard or semistandard Young tableaux.
- `row` reads one row at a time. `col` reads columns from left to right and each column from top to bottom, skipping missing cells in ragged shapes.
- `coord` emits every cell as a 1-based `(row,column) : value` entry. Coordinates use the conventional matrix order and entries are listed row-major: top to bottom, then left to right within each row.
- Surface formats are strictly canonical and use ASCII punctuation and spaces. Groups are separated by ` | `; coordinate tuples have no internal spaces and use exactly `) : value` after each tuple.
- Surface notation always writes ordinary decimal numbers, such as `(1,2) : 3`. Names such as `n1`, `n2`, and `n3` are internal tokenizer symbols and must not be written in CLI input.
- Synthetic data is deduplicated and split by the underlying tableau before row/column/coordinate forms are expanded. Different textual representations of the same tableau therefore cannot leak across train, validation, and test splits.
- The RSK task uses standard row insertion in English/matrix orientation and outputs `P` only. Permutations with the same `P` are kept in the same split, preventing Knuth-equivalent inputs from leaking an identical target across train and test.

## Four text formats

The same tableau from the screenshot is:

```text
2 3 5
1 4
```

The repository supports four canonical surface forms: one raw form and three human-readable forms.

```text
raw: [YT start] 2 3 5 | 1 4 [YT end]
row: [YT row start] 2 3 5 | 1 4 [YT row end]
col: [YT col start] 2 1 | 3 4 | 5 [YT col end]
coord: [YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | (2,1) : 1 | (2,2) : 4 [YT coord end]
```

The bodies of `raw` and `row` are both row-major, but they use different markers. The three groups in `col` represent columns `(2, 1)`, `(3, 4)`, and `(5)`. In `coord`, `(1,1)` identifies the top-left cell and `(2,2)` identifies the second row's second cell. Coordinate entries must appear in canonical row-major order. All four formats can be parsed losslessly into the same tableau.

## Hand-written tokenizer

The notation vocabulary contains 70 fixed tokens whose IDs are not learned. IDs `0..62` retain their original meanings; coordinate support is append-only at IDs `63..69` so the existing token IDs do not move. RSK models explicitly use a 72-token superset that appends the permutation markers at IDs `70..71`.

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
| 63 | `TO_COORD` | Control token requesting coordinate output |
| 64 | `x7` | `[YT coord start]` |
| 65 | `x8` | `[YT coord end]` |
| 66 | `lparen` | `(` |
| 67 | `comma` | `,` |
| 68 | `rparen` | `)` |
| 69 | `colon` | `:` |
| 70 | `x9` | `[perm start]` (RSK vocabulary only) |
| 71 | `x10` | `[perm end]` (RSK vocabulary only) |

For example, row notation is encoded as:

```text
BOS x1 s n2 s n3 s n5 s x s n1 s n4 s x2 EOS
```

Coordinate surface text still contains decimal numbers. For example, the first two entries of the coordinate form above are encoded as:

```text
BOS x7 s lparen n1 comma n1 rparen s colon s n2 s x s lparen n1 comma n2 rparen s colon s n3 ... s x8 EOS
```

The tokenizer strictly parses canonical notation before encoding it. Unknown markers, malformed or misordered coordinates, noncanonical whitespace, leading zeroes, and numbers outside `1..50` are rejected instead of being silently normalized.

### Why forward conversion needs a control token

The same raw input has three valid targets:

```text
raw + TO_ROW -> row notation
raw + TO_COL -> col notation
raw + TO_COORD -> coord notation
```

Without a control token, an identical source would map to three different targets and the model would not know which form to generate. During encoding, the control token appears immediately after `BOS`:

```text
BOS TO_ROW x5 ... x6 EOS
BOS TO_COL x5 ... x6 EOS
BOS TO_COORD x5 ... x6 EOS
```

The reverse model does not need an additional control token because the first input marker already disambiguates the format: row notation begins with `x1`, column notation begins with `x3`, coordinate notation begins with `x7`, and all three map to raw notation.

## Installation

Python 3.10 or later and PyTorch 2.1 or later are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you need a specific CUDA build, install the appropriate PyTorch package for your platform first, then run `python -m pip install -e .`. The installation exposes the notation commands `yt-convert`, `yt-train`, `yt-infer`, `yt-evaluate`, and `yt-ood`, plus `yt-rsk`, `yt-rsk-train`, `yt-rsk-infer`, and `yt-rsk-evaluate` for the permutation task.

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

Convert raw notation to coordinate notation:

```bash
yt-convert '[YT start] 2 3 5 | 1 4 [YT end]' --to coord
```

Output:

```text
[YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | (2,1) : 1 | (2,2) : 4 [YT coord end]
```

Convert coordinate notation back to raw notation:

```bash
yt-convert '[YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | (2,1) : 1 | (2,2) : 4 [YT coord end]' --to raw
```

Valid values for `--to` are `raw`, `row`, `col`, and `coord`.

## RSK: permutation to insertion tableau

The RSK input surface is a nonempty permutation of exactly `1..n`, with
`n <= 50`:

```text
[perm start] 3 5 1 4 2 [perm end]
```

Values are inserted from left to right. In each row, the first existing value
strictly greater than the carried value is replaced and bumped to the next row;
if none is greater, the carried value is appended. The result is the insertion
tableau `P` in raw YT notation:

```text
[YT start] 1 2 | 3 4 | 5 [YT end]
```

Compute it exactly, without a checkpoint:

```bash
yt-rsk '[perm start] 3 5 1 4 2 [perm end]'
```

Only `P` is produced. There is intentionally no `YT -> permutation` task:
recovering a permutation requires the recording tableau `Q`, and different
permutations can have the same `P`.

Train the separate RSK model:

```bash
yt-rsk-train \
  --output-dir checkpoints/rsk \
  --device auto \
  --num-permutations 80000 \
  --min-length 1 \
  --max-length 20 \
  --split-seed 43 \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --patience 10 \
  --val-exact-limit 512 \
  --d-model 256 \
  --nhead 8 \
  --num-layers 4 \
  --dim-feedforward 1024 \
  --max-seq-len 128 \
  --seed 42
```

The recorded run was stopped during epoch 11, so its saved partial result is
the best checkpoint from epoch 10 rather than a completed 50-epoch run. The
generator balances lengths as far as the finite small-`n` permutation
spaces allow, includes increasing/decreasing edge cases, and splits entire
`P`-groups together. The saved file is
`checkpoints/rsk/perm_to_yt.pt`.

Run learned inference and held-out evaluation with the RSK-specific commands:

```bash
yt-rsk-infer \
  --checkpoint checkpoints/rsk/perm_to_yt.pt \
  --text '[perm start] 3 5 1 4 2 [perm end]' \
  --device auto

yt-rsk-evaluate \
  --checkpoint checkpoints/rsk/perm_to_yt.pt \
  --device auto \
  --batch-size 128
```

The evaluator reports greedy surface and semantic exact match, parse validity,
shape and value-content accuracy, teacher-forced token metrics, and results by
permutation length. Exact training configuration, artifact hash, and measured
results are recorded in
[`RSK_TRAINING_RESULTS.md`](RSK_TRAINING_RESULTS.md).

## Training both models

### Full training

The following command uses the main default configuration: 4,000 unique tableaux, a `0.8/0.1/0.1` split, row, column, and coordinate targets, and at most 15 epochs. `auto` prefers CUDA, then MPS, and finally falls back to CPU.

```bash
yt-train \
  --direction both \
  --output-dir checkpoints/all-formats \
  --device auto \
  --num-tableaux 4000 \
  --split-ratios 0.8 0.1 0.1 \
  --human-kinds row col coord \
  --epochs 15 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --patience 5 \
  --val-exact-limit 256 \
  --max-seq-len 256 \
  --seed 42
```

Training saves the best validation checkpoint for each direction:

```text
checkpoints/all-formats/yt_to_human.pt
checkpoints/all-formats/human_to_yt.pt
```

The default model uses `d_model=64`, 4 attention heads, 2 encoder layers, 2 decoder layers, a feed-forward dimension of 128, dropout 0.1, and a maximum sequence length of 256. Coordinate notation is substantially longer than row or column notation: a 20-cell coordinate target can occupy 243 tokens including `BOS` and `EOS`, so the previous limit of 128 is not sufficient. Because the input and output use the same fixed vocabulary, source embeddings, target embeddings, and output-projection token weights are shared by default. Use `--no-tie-embeddings` to disable sharing. Each checkpoint stores its direction, model configuration, fixed vocabulary, best epoch, metrics, and synthetic-data configuration so that evaluation can reconstruct the exact held-out split.

To train only one direction, replace `--direction both` with `yt_to_human` or `human_to_yt`.

The trained coordinate-only checkpoints in this workspace use 8,000 tableaux,
a larger 128-dimensional model, and `--human-kinds coord`. See
[`COORD_TRAINING_RESULTS.md`](COORD_TRAINING_RESULTS.md) for the exact
reproduction command, hashes, and held-out metrics.

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
  --human-kinds row col coord \
  --epochs 2 \
  --batch-size 32 \
  --patience 0 \
  --val-exact-limit 24 \
  --d-model 32 \
  --nhead 4 \
  --num-layers 1 \
  --dim-feedforward 64 \
  --dropout 0 \
  --max-seq-len 128 \
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

Convert raw notation to coordinate notation:

```bash
yt-infer \
  --checkpoint checkpoints/coord/yt_to_human.pt \
  --text '[YT start] 2 3 5 | 1 4 [YT end]' \
  --style coord \
  --device auto
```

Convert human-readable column notation back to raw notation:

```bash
yt-infer \
  --checkpoint checkpoints/human_to_yt.pt \
  --text '[YT col start] 2 1 | 3 4 | 5 [YT col end]' \
  --device auto
```

Convert human-readable coordinate notation back to raw notation:

```bash
yt-infer \
  --checkpoint checkpoints/coord/human_to_yt.pt \
  --text '[YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | (2,1) : 1 | (2,2) : 4 [YT coord end]' \
  --device auto
```

For a `human_to_yt` checkpoint, the program determines row, column, or coordinate input from the marker, so `--style` does not affect reverse conversion. The optional `--max-new-tokens N` argument can reduce the generation limit; by default, inference uses the checkpoint's model sequence limit.

Inference also checks the checkpoint's recorded `human_kinds`. The coordinate-only
models reject row/column requests, and the legacy row/column models reject
coordinate requests, instead of attempting an untrained conversion.

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
  --checkpoint checkpoints/coord/yt_to_human.pt \
  --checkpoint checkpoints/coord/human_to_yt.pt \
  --device auto \
  --batch-size 64 \
  --round-trip-limit 100
```

Pass `--checkpoint` once or twice. When supplied twice, the checkpoints must
represent different directions and use the same vocabulary; pair checkpoints
from the same directory rather than mixing a legacy 63-token model with a
coordinate-aware 70-token model. JSON output for each direction contains:

- `loss` and teacher-forced `token_accuracy`.
- Greedy-decoding `exact_match`.
- `semantic_accuracy`, which requires both the parsed tableau and target format to be correct.
- `invalid_output_rate` for outputs that cannot be parsed or do not emit `EOS`.

When both directions are available, the report also contains
`round_trip.exact_match` and `invalid_pipeline_rate`. Round trips use the human
styles shared by the two checkpoints, so coordinate-only and legacy
row/column-only pairs are both handled correctly. `--limit N` limits held-out
examples per direction, while `--round-trip-limit N` limits the number of
unique tableaux used for round-trip evaluation.

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
| `yt_transformer/notation.py` | `Tableau` and strict parsing/serialization of all four formats |
| `yt_transformer/tokenizer.py` | The compatible 63/70/72-token hand-written vocabularies and encode/decode logic |
| `yt_transformer/data.py` | Reproducible synthetic tableaux, paired examples, splits, Dataset, and collation |
| `yt_transformer/model.py` | Small batch-first encoder-decoder Transformer |
| `yt_transformer/train.py` | Training, validation, and early stopping for both directions |
| `yt_transformer/checkpoint.py` | Atomic, versioned checkpoint saving and loading |
| `yt_transformer/convert.py` | Deterministic model-free format conversion |
| `yt_transformer/infer.py` | Single-example checkpoint inference |
| `yt_transformer/evaluate.py` | Held-out metrics and two-model round-trip evaluation |
| `yt_transformer/ood.py` | Permutation and length-extrapolation tests beyond training cell counts |
| `yt_transformer/rsk.py` | Strict permutation format and exact standard row-insertion oracle |
| `yt_transformer/rsk_data.py` | Length-aware RSK examples, P-group splits, Dataset, and collation |
| `yt_transformer/rsk_convert.py` | Deterministic permutation-to-`P` conversion |
| `yt_transformer/rsk_train.py` | Training and validation for `perm_to_yt` |
| `yt_transformer/rsk_infer.py` | Single-permutation learned inference |
| `yt_transformer/rsk_evaluate.py` | Held-out RSK metrics, including per-length results |
| `yt_transformer/runtime.py` | Random seeds and CPU/CUDA/MPS device selection |
| `tests/` | Standard-library unit test suite |
| `TRAINING_RESULTS.md` | Earlier row/column checkpoint configuration and metrics |
| `COORD_TRAINING_RESULTS.md` | Coordinate-only checkpoint configuration and metrics |
| `RSK_TRAINING_RESULTS.md` | Permutation-to-insertion-tableau checkpoint configuration and metrics |

## Current limitations

- The project only processes canonical **plain text**. It cannot directly read grid images, chat screenshots, OCR output, or LaTeX. Such inputs require a separate vision/OCR/parser stage that produces raw notation.
- The notation and permutation-to-RSK tasks are implemented. **Large-number representation and `A + B = C` addition** are not implemented, and no base-50 or bijective-base-50 arithmetic model has been trained.
- The vocabulary only covers integers and coordinate indices `1..50`; models should only be used within the shape and sequence-length limits configured during training.
- Synthetic fillings do not satisfy the uniqueness or monotonicity rules of standard or semistandard Young tableaux. If those mathematical constraints are required, replace the data-generation rules and retrain.
- Neural models cannot guarantee valid or correct output. Use `yt-convert` or `yt-rsk` when conversion must be exact, and use the held-out evaluators when measuring learned-model behavior.
- Checkpoints are loaded with PyTorch's `weights_only=True` and validated for version, vocabulary, and configuration, but only checkpoints created locally or obtained from a trusted source should be loaded. Do not treat arbitrary uploaded files as safe inputs.
