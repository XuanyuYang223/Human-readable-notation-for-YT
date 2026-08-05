# Length-extrapolation results

The trained models only saw tableaux with at most 20 cells. To measure whether they generalize to longer permutations, we used seed 2026 to generate 100 OOD tableaux at each length and evaluated both conversion directions and both human-readable styles.

For 20, 21, 30, 40, and 50 cells, every tableau contains distinct values sampled from `1..50`. A 54-cell tableau cannot contain 54 distinct values under the fixed vocabulary, so that case shuffles `1..50` plus four repeated values and serves only as a sequence-length stress test.

## Important distribution difference

This is not a single-variable experiment that changes only length. Training fillings sample each cell independently from `1..50` **with replacement**, whereas the 20-to-50-cell OOD fillings are permutations sampled **without replacement**. The OOD experiment also uses fixed compact shapes.

Among the 6,400 training tableaux generated with seed 42:

- 353 tableaux contain 20 cells.
- Only 5 of those contain 20 distinct values.
- 19 use shape `(4,4,4,4,4)`.
- No training example has both that shape and all-unique values.

The results below therefore measure OOD generalization under a combination of length, shape, and filling changes. The entire drop cannot be attributed to length alone.

## Exact-match results

| Entries | Shape | Unique values | YT to row | YT to col | Row to YT | Col to YT |
|---:|---|:---:|---:|---:|---:|---:|
| 20 | `(4,4,4,4,4)` | Yes | 100% | 83% | 99% | 75% |
| 21 | `(5,5,5,5,1)` | Yes | 92% | 5% | 0% | 0% |
| 30 | `(6,6,6,6,6)` | Yes | 0% | 0% | 0% | 0% |
| 40 | `(8,8,8,8,8)` | Yes | 0% | 0% | 0% | 0% |
| 50 | `(10,10,10,10,10)` | Yes | 0% | 0% | 0% | 0% |
| 54 | `(8,8,8,8,8,8,6)` | No | 0% | 0% | 0% | 0% |

Although teacher-forced token accuracy remains between 95% and 99% for parts of the 21-cell evaluation, whole-sequence exact accuracy falls sharply. All four tasks reach 0% at 30 cells and above. We can conclude that the models **do not generalize to the longer, all-unique, fixed-compact-shape permutation distribution**. This experiment alone cannot determine whether length, unique-value filling, shape, or their interaction is the dominant cause.

This finding does not conflict with the approximately 99.5% ordinary held-out exact match. The ordinary held-out data comes from the same 1-to-20-cell, with-replacement filling distribution as training, whereas this experiment deliberately changes multiple factors.

## Reproduction

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

1. Run a controlled ablation at a fixed 20-cell shape, comparing with-replacement fillings against without-replacement permutations.
2. Hold the filling rule fixed and compare 20 against 21 cells to isolate the length effect.
3. If permutations are the intended task, change the training generator to select `N` and then shuffle `1..N`, rather than sampling every cell independently with replacement.
4. Expand training to 1-to-50 cells with curriculum learning or length-balanced sampling, and report validation/test metrics in separate length and shape buckets.
5. Increase the maximum model sequence length to 256 and compare absolute positions against relative positions, RoPE, or ALiBi.
6. If the task requires more than 50 distinct entries, replace the atomic `n1..n50` vocabulary with a digit-level or otherwise compositional number tokenizer. Simply adding `n51` would not make an existing model understand the new token.
