# Trained checkpoint results

训练日期：2026-08-05。运行环境：NVIDIA GeForce RTX 5070 12GB、PyTorch 2.11.0 + CUDA 12.8。

本 workspace 中已经生成两个 checkpoint（`checkpoints/` 被 Git 忽略，避免误提交二进制文件）：

| checkpoint | 大小 | 最佳 epoch | 训练时 validation-subset exact |
|---|---:|---:|---:|
| `checkpoints/yt_to_human.pt` | 2.7 MB | 37 | 99.6% |
| `checkpoints/human_to_yt.pt` | 2.7 MB | 40 | 100.0% |

SHA-256：

```text
81db3db7977b61ae8af014d210a6c5b4e47e3ef579f6568b706b36eb7a2fcc0e  checkpoints/yt_to_human.pt
6d1e3d2d762fdf4b3d46088f19c007bee1459fbe563b7ae279be8f3fc1045cd4  checkpoints/human_to_yt.pt
```

## Training configuration

- 8,000 个唯一合成 Tableau，train/val/test 为 0.8/0.1/0.1；
- row 与 col 两种 human-readable target；
- 最多 5 行、8 列、20 个格子，值域 1..50；
- 每个模型 671,167 个参数；
- `d_model=128`、8 heads、2 层 encoder、2 层 decoder、FFN 256；
- shared source/target/output token weights，dropout 0.1；
- batch size 256，AdamW，learning rate 8e-4，最多 40 epochs；
- seed 42，训练设备 `cuda:0`。

复现命令：

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

这里的每个方向都使用全部 1,600 条 held-out examples，不是训练日志中的 256 条 autoregressive validation 子集。

| direction | exact match | semantic accuracy | token accuracy | invalid output |
|---|---:|---:|---:|---:|
| YT → human | 99.5625% | 99.5625% | 99.9845% | 0.1250% |
| human → YT | 99.4375% | 99.4375% | 99.9814% | 0.0625% |

在 100 个唯一 Tableau、row/col 共 200 次 `raw → human → raw` round trip 上：

- exact match：99.5%；
- invalid pipeline rate：0%。

## Screenshot example

模型实际输出：

```text
input:  [YT start] 2 3 5 | 1 4 [YT end]
row:    [YT row start] 2 3 5 | 1 4 [YT row end]
col:    [YT col start] 2 1 | 3 4 | 5 [YT col end]
back:   [YT start] 2 3 5 | 1 4 [YT end]
```
