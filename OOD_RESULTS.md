# Length extrapolation results

正式模型训练时只见过最多 20 格的 Tableau。为判断模型是否能泛化到更长的 permutation，使用 seed 2026 为每个长度生成 100 个 OOD Tableau，并分别测试两个方向和两种 human-readable style。

对于 20、21、30、40、50 格，每个 Tableau 的值互不重复，均从 `1..50` 随机抽取。54 格超过固定词表可提供的不同值数量，因此使用打乱后的 `1..50` 加 4 个重复值，只测试序列长度。

## Important distribution difference

这个实验不是只改变长度的单变量实验。训练数据的 filling 是每格独立从 `1..50` **有放回采样**，而本实验的 20–50 格 filling 是**无放回 permutation**；同时还固定使用紧凑 shape。

本次 seed=42 的 6,400 个训练 Tableau 中：

- 20 格 Tableau 有 353 个；
- 其中只有 5 个恰好 20 个值全不重复；
- shape `(4,4,4,4,4)` 有 19 个；
- “该 shape 且所有值不重复”的训练样例为 0 个。

因此下面结果衡量的是长度、shape 和 filling 三者组合的 OOD 泛化，不能把全部下降单独归因于长度。

## Exact-match results

| entries | shape | unique values | YT → row | YT → col | row → YT | col → YT |
|---:|---|:---:|---:|---:|---:|---:|
| 20 | `(4,4,4,4,4)` | yes | 100% | 83% | 99% | 75% |
| 21 | `(5,5,5,5,1)` | yes | 92% | 5% | 0% | 0% |
| 30 | `(6,6,6,6,6)` | yes | 0% | 0% | 0% | 0% |
| 40 | `(8,8,8,8,8)` | yes | 0% | 0% | 0% | 0% |
| 50 | `(10,10,10,10,10)` | yes | 0% | 0% | 0% | 0% |
| 54 | `(8,8,8,8,8,8,6)` | no | 0% | 0% | 0% | 0% |

21 格时虽然部分 teacher-forced token accuracy 仍在 95–99%，整串 exact accuracy 已经断崖式下降；到 30 格及以上四个任务全部为 0%。可以确定的是：模型**没有泛化到更长、值全唯一、固定紧凑 shape 的 permutation 分布**。仅凭这一组实验，还不能区分主要原因是长度、唯一值 filling、shape，还是三者交互。

这个结论不与普通 held-out 的约 99.5% exact 冲突：普通 held-out 与训练集来自相同的 1–20 格、有放回 filling 分布，而这里刻意改变了多个因素。

## Reproduce

```bash
yt-ood \
  --yt-to-human checkpoints/yt_to_human.pt \
  --human-to-yt checkpoints/human_to_yt.pt \
  --entries 20 21 30 40 50 54 \
  --samples 100 \
  --batch-size 128 \
  --seed 2026 \
  --device cuda:0
```

## Recommended next experiment

1. 先做受控 ablation：固定 20 格 shape，分别比较有放回 filling 与无放回 permutation。
2. 再固定 filling 规则，比较 20 格与 21 格，单独测量长度效应。
3. 如果目标本来就是 permutation，应把训练生成器改为先选 `N`、再打乱 `1..N`，而不是独立有放回采样。
4. 把训练分布扩展为 1–50 格，并按长度做 curriculum 或均衡采样；validation/test 按长度和 shape 分桶。
5. 将模型最大序列长度提高到 256，并比较 absolute position 与 relative position、RoPE 或 ALiBi。
6. 如果目标包括 51 以上的互不相同 entry，`n1..n50` 原子词表必须改为 digit-level/可组合数字 tokenizer；简单添加 `n51` 不会让旧模型自动理解新 token。
