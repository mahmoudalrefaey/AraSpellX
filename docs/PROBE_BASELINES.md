# Probe baselines

Reference results of `evaluate_probes.py` (default seed and sizes, probe
sentences from the first 20,000 rows of `data/dataset/test.csv`). Compare new
checkpoints against these numbers with the same command:

    python evaluate_probes.py --checkpoint <checkpoint>

- **Old model**: `checkpoint_epoch_2_step_648966.pt`, 3 full epochs on
  `distorted_0.1` (random typos only, no spelling habits, no padding masks).
  The file itself has been deleted; these numbers are kept for comparison.
- **Step 35k**: `checkpoint_epoch_0_step_35000.pt`, 16% of epoch 0 on the first
  `distorted_rw` data (typo ratios 0/0.02/0.05/0.1, five habits, padding masks),
  stopped at the peak learning rate.

The habit probes use all habits of `get_habits`, including the two added after
both models were trained (ء after alef on a ya seat, dropped alef after waw).

## Old model (3 epochs, distorted_0.1)

```text
probe                                 n  fixed   left  wrong words±  exact  damage
1 typo                              150  0.693  0.140  0.147  0.020  0.600  0.0241
2 typos in one word                 150  0.387  0.100  0.447  0.067  0.340  0.0238
3 typos in one word                 150  0.300  0.067  0.533  0.100  0.240  0.0304
1 typo in each of 3 words           150  0.360  0.007  0.547  0.087  0.313  0.0232
habits only                         150  0.580  0.280  0.093  0.047  0.007  0.3450
habits + 2 typos in one word        150  0.300  0.107  0.527  0.067  0.027  0.3268
single: deleted letter              120  0.458  0.325  0.192  0.025  0.375  0.0225
single: doubled letter              120  0.800  0.033  0.125  0.042  0.625  0.0319
single: keyboard substitution       120  0.542  0.217  0.200  0.042  0.408  0.0340
single: swapped letters             120  0.717  0.133  0.117  0.033  0.600  0.0260
single: inserted letter             120  0.867  0.042  0.058  0.033  0.725  0.0220
fixed/left/wrong: corrupted word(s) restored / copied unchanged / changed to something else; words±: word count changed; damage: share of the other, correct words that were changed

Benchmark data/benchmarks/real_world.csv (27 rows)
Input (no correction): CER=0.0679 WER=0.3838 exact=0.000
Model output:          CER=0.0712 WER=0.3454 exact=0.000
```

## Step 35k (distorted_rw, first version)

```text
probe                                 n  fixed   left  wrong words±  exact  damage
1 typo                              150  0.273  0.573  0.127  0.027  0.240  0.0236
2 typos in one word                 150  0.033  0.527  0.387  0.053  0.033  0.0183
3 typos in one word                 150  0.040  0.373  0.547  0.040  0.033  0.0082
1 typo in each of 3 words           150  0.020  0.280  0.627  0.073  0.020  0.0175
habits only                         150  0.900  0.027  0.053  0.020  0.527  0.0642
habits + 2 typos in one word        150  0.113  0.353  0.513  0.020  0.093  0.0628
single: deleted letter              120  0.008  0.908  0.067  0.017  0.008  0.0135
single: doubled letter              120  0.517  0.408  0.075  0.000  0.475  0.0156
single: keyboard substitution       120  0.083  0.658  0.217  0.042  0.075  0.0175
single: swapped letters             120  0.117  0.750  0.125  0.008  0.108  0.0068
single: inserted letter             120  0.283  0.592  0.108  0.017  0.258  0.0119
fixed/left/wrong: corrupted word(s) restored / copied unchanged / changed to something else; words±: word count changed; damage: share of the other, correct words that were changed

Benchmark data/benchmarks/real_world.csv (27 rows)
Input (no correction): CER=0.0679 WER=0.3838 exact=0.000
Model output:          CER=0.0144 WER=0.0789 exact=0.259
```
