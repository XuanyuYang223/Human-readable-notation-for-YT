# Human-readable notation for YT

这个项目把截图中的 **YT（Young tableau，杨表）格式转换任务**实现为一个完整、可复现的小型实验：用固定的人工 tokenizer 和合成数据，分别训练两个 encoder-decoder Transformer：

- `yt_to_human`：从 raw YT 文本生成 row 或 col human-readable notation；
- `human_to_yt`：从 row 或 col human-readable notation 还原 raw YT 文本。

项目同时提供一个不依赖模型的确定性转换器。确定性转换器是数据标签与正确性的 oracle；Transformer 则用于完成“训练一个简单模型学习该转换”的实验目标。

当前 workspace 已训练好两个正式 checkpoint；训练配置、hash 和完整 held-out 指标见 [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md)。

Permutation OOD 实验表明模型无法泛化到“更长且值全唯一”的组合分布；该实验同时改变了长度、shape 与 filling，完整结果和受控实验建议见 [`OOD_RESULTS.md`](OOD_RESULTS.md)。

## 范围与明确假设

- `YT` 在本仓库中指 Young tableau，不指 YouTube。
- 模型输入是**规范化纯文本**，不是图片、截图、手写表格或 LaTeX。
- Tableau 内部表示为从上到下排列的行；每行从左到右读取，图形左对齐。
- 行长度必须非递增，例如 `(3, 2, 1)` 合法，`(2, 3)` 不合法。
- 每个格子的值都是 `1..50` 的整数。当前任务不要求值唯一，也不强加行或列的数值单调性，因此它不是 standard/semi-standard Young tableau 的生成器。
- `row` 逐行读取；`col` 从左到右逐列读取，每列从上到下，并跳过 ragged shape 中不存在的格子。
- 表面格式是严格 canonical 的：使用 ASCII 空格，任意两个表面 token 之间恰好一个空格，分组符写作 ` | `。
- 合成数据先按 Tableau 本体去重和切分，再展开 row/col 形式，因此同一个 Tableau 的不同文本形式不会泄漏到不同的 train/val/test split。

## 三种文本格式

截图中的同一个 Tableau：

```text
2 3 5
1 4
```

在仓库中有三种 canonical surface form：

```text
raw: [YT start] 2 3 5 | 1 4 [YT end]
row: [YT row start] 2 3 5 | 1 4 [YT row end]
col: [YT col start] 2 1 | 3 4 | 5 [YT col end]
```

`raw` 和 `row` 的 body 都按行排列，但 marker 不同；`col` 的三个分组分别是列 `(2, 1)`、`(3, 4)` 和 `(5)`。三种格式都可以无损解析回同一个 Tableau。

## 人工 tokenizer

词表固定为 63 个 token，ID 不通过训练学习：

| ID | token | 对应表面文本或作用 |
|---:|---|---|
| 0 | `PAD` | batch padding |
| 1 | `BOS` | 序列开始 |
| 2 | `EOS` | 序列结束 |
| 3 | `TO_ROW` | forward 输出 row 的控制 token |
| 4 | `TO_COL` | forward 输出 col 的控制 token |
| 5 | `x1` | `[YT row start]` |
| 6 | `x2` | `[YT row end]` |
| 7 | `x3` | `[YT col start]` |
| 8 | `x4` | `[YT col end]` |
| 9 | `x5` | `[YT start]` |
| 10 | `x6` | `[YT end]` |
| 11 | `s` | 一个 ASCII 空格 |
| 12 | `x` | `|` |
| 13..62 | `n1`..`n50` | 整数 `1`..`50`，其中 `nK` 的 ID 为 `K + 12` |

例如 row 文本会被编码为：

```text
BOS x1 s n2 s n3 s n5 s x s n1 s n4 s x2 EOS
```

Tokenizer 会先严格解析 canonical notation；未知 marker、非规范空白、前导零和 `1..50` 之外的数字都会被拒绝，而不是被静默修正。

### 为什么 forward 需要 control token

同一个 raw 输入需要支持两个合法目标：

```text
raw + TO_ROW -> row notation
raw + TO_COL -> col notation
```

如果不给控制 token，完全相同的 source 会同时对应两个不同 target，模型无法知道本次应生成哪一种形式。实际编码时控制 token 位于 `BOS` 后：

```text
BOS TO_ROW x5 ... x6 EOS
BOS TO_COL x5 ... x6 EOS
```

反向模型不需要额外控制 token，因为输入自身的首 marker 已经消歧：row 以 `x1` 开始，col 以 `x3` 开始；二者都应输出 raw notation。

## 安装

需要 Python 3.10 或更高版本，以及 PyTorch 2.1 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

如果需要特定 CUDA 版本，请先按 PyTorch 官方方式安装匹配平台的 PyTorch，再执行 `python -m pip install -e .`。安装后可使用 `yt-convert`、`yt-train`、`yt-infer`、`yt-evaluate` 和 `yt-ood` 命令。

## 确定性转换

确定性转换不加载 checkpoint，也不会产生模型误差，适合检查格式、生成标签和调试 tokenizer。

raw 转 col：

```bash
yt-convert '[YT start] 2 3 5 | 1 4 [YT end]' --to col
```

输出：

```text
[YT col start] 2 1 | 3 4 | 5 [YT col end]
```

col 转回 raw：

```bash
yt-convert '[YT col start] 2 1 | 3 4 | 5 [YT col end]' --to raw
```

`--to` 的可选值为 `raw`、`row`、`col`。

## 训练两个模型

### 正式训练

下面的命令使用主要默认配置：4,000 个唯一 Tableau、`0.8/0.1/0.1` split、row 与 col 两种 target、最多 15 epochs。`auto` 会依次优先选择 CUDA、MPS，最后回退到 CPU。

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

训练会分别保存验证集表现最好的 checkpoint：

```text
checkpoints/yt_to_human.pt
checkpoints/human_to_yt.pt
```

默认模型为 `d_model=64`、4 heads、2 层 encoder 和 2 层 decoder、feed-forward dimension 128、dropout 0.1、最大序列长度 128。由于输入输出使用同一固定词表，默认共享 source embedding、target embedding 和 output projection 的 token 权重；可用 `--no-tie-embeddings` 关闭。checkpoint 包含方向、模型配置、固定词表、最佳 epoch、指标和合成数据配置；评估程序据此重建完全相同的 held-out split。

如只需一个方向，把 `--direction both` 改为 `yt_to_human` 或 `human_to_yt`。

### 快速 smoke training

下面的 CPU 命令用于验证端到端流程是否能运行，不代表正式训练质量：

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

`--patience 0` 表示关闭 early stopping。训练会固定 Python 与 PyTorch 的随机种子；合成数据和 split 是确定的，但不同硬件或底层 PyTorch kernel 上的浮点训练结果不保证逐 bit 相同。

## 两个方向的模型推理

以下命令使用模型的 greedy decoding。快速 smoke checkpoint 可能尚未学会任务；若模型没有输出合法 marker 或未在长度限制内输出 `EOS`，命令会明确报错并显示生成 token。

raw 转 row：

```bash
yt-infer \
  --checkpoint checkpoints/yt_to_human.pt \
  --text '[YT start] 2 3 5 | 1 4 [YT end]' \
  --style row \
  --device auto
```

raw 转 col：

```bash
yt-infer \
  --checkpoint checkpoints/yt_to_human.pt \
  --text '[YT start] 2 3 5 | 1 4 [YT end]' \
  --style col \
  --device auto
```

human-readable col 转回 raw：

```bash
yt-infer \
  --checkpoint checkpoints/human_to_yt.pt \
  --text '[YT col start] 2 1 | 3 4 | 5 [YT col end]' \
  --device auto
```

对于 `human_to_yt` checkpoint，程序会从输入 marker 自动判断 row/col，`--style` 不参与反向转换。可选的 `--max-new-tokens N` 用于收紧生成上限；默认使用 checkpoint 的模型序列上限。

## 评估 checkpoint

单 checkpoint 评估：

```bash
yt-evaluate \
  --checkpoint checkpoints/yt_to_human.pt \
  --device auto \
  --batch-size 64
```

双 checkpoint 评估与 raw → human → raw round trip：

```bash
yt-evaluate \
  --checkpoint checkpoints/yt_to_human.pt \
  --checkpoint checkpoints/human_to_yt.pt \
  --device auto \
  --batch-size 64 \
  --round-trip-limit 100
```

`--checkpoint` 必须传一次或两次；传两次时必须分别是两个不同方向。输出为 JSON，单方向包含：

- `loss` 和 teacher-forced `token_accuracy`；
- greedy decoding 的 `exact_match`；
- 解析后 Tableau 与目标格式都正确的 `semantic_accuracy`；
- 无法解析或缺少 `EOS` 的 `invalid_output_rate`。

两个方向齐全时还会输出 `round_trip.exact_match` 和 `invalid_pipeline_rate`。`--limit N` 可以限制每个方向的 held-out 样例数；`--round-trip-limit N` 限制 round-trip 使用的唯一 Tableau 数。

## 运行测试

测试全部使用标准库 `unittest`：

```bash
python -m unittest discover -s tests -v
```

测试覆盖 canonical format/parse、固定 token ID、合成数据复现与 split 防泄漏、模型 shape 与 decoding、checkpoint 校验、确定性转换、训练/推理/评估核心路径。

## 超出训练长度的 permutation 测试

正式模型只在最多 20 格的 Tableau 上训练。下面的命令以 20 格为基线，测试 21、30、40、50 格的随机无重复 permutation，以及 54 格的重复值长度压力样例：

```bash
yt-ood \
  --yt-to-human checkpoints/yt_to_human.pt \
  --human-to-yt checkpoints/human_to_yt.pt \
  --entries 20 21 30 40 50 54 \
  --samples 100 \
  --device cuda:0
```

因为当前人工词表严格限定为 `n1..n50`，50 是互不相同 entry permutation 的最大长度。54 格 case 会重复部分 `1..50` token，只用于测试序列长度；如需 51 个以上互不相同的值，应改用可组合的 digit tokenizer 并重新训练。

## 文件结构

| 路径 | 作用 |
|---|---|
| `yt_transformer/notation.py` | `Tableau`、三种格式的严格解析与序列化 |
| `yt_transformer/tokenizer.py` | 63-token 人工词表及 encode/decode |
| `yt_transformer/data.py` | 可复现合成 Tableau、双方向样例、split、Dataset/collate |
| `yt_transformer/model.py` | 小型 batch-first encoder-decoder Transformer |
| `yt_transformer/train.py` | 两个方向的训练、验证与 early stopping |
| `yt_transformer/checkpoint.py` | 带版本和元数据的原子化 checkpoint 保存/加载 |
| `yt_transformer/convert.py` | 不依赖模型的确定性格式转换 |
| `yt_transformer/infer.py` | 单条样例的 checkpoint 推理 |
| `yt_transformer/evaluate.py` | held-out 指标与双模型 round-trip 评估 |
| `yt_transformer/ood.py` | 超出训练格数的 permutation/长度外推测试 |
| `yt_transformer/runtime.py` | 随机种子与 CPU/CUDA/MPS 设备选择 |
| `tests/` | 标准库 unittest 测试集 |

## 当前限制

- 本项目只处理 canonical **纯文本**。它不能直接读取图片中的方格、聊天截图、OCR 结果或 LaTeX；这些输入需要先由独立的视觉/OCR/解析步骤转换成 raw notation。
- 这只是截图任务的第一部分。截图中的 **Task 2 大数表示与 `A + B = C` 加法**尚未实现，也未训练 base-50/bijective-base-50 算术模型。
- 词表只覆盖整数 `1..50`，模型只应在训练时配置的 shape 与序列长度范围内使用。
- 合成 filling 不满足 standard Young tableau 的唯一性和数值单调约束；如后续确认需要数学意义上的 standard/semi-standard tableau，应更换数据生成规则并重新训练。
- 神经模型不能保证每次都生成合法或正确结果。需要保证正确转换时，请使用 `yt-convert`；需要衡量模型能力时，请使用 held-out 和 round-trip 评估。
- checkpoint 使用 PyTorch 的 `weights_only=True` 加载并校验版本、词表与配置，但仍应只加载自己训练或可信来源提供的文件；不要把任意不可信上传文件当作安全输入。
