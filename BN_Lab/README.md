# AI Laboratory: Bayesian Networks and Autoregressive Language Models

Pure-Python (standard library only) implementations of a first-order and a
second-order autoregressive language model, viewed as Bayesian networks.

## Files

| File | What it does |
|------|--------------|
| `first_order.py` | First-order model `P(Xt \| Xt-1)`: bigram counts, CPT, next-word prediction, sampling and greedy generation, normalisation test |
| `second_order.py` | Second-order model `P(Xt \| Xt-2, Xt-1)`: trigram counts, same features, plus the Part XIII first-vs-second-order comparison (imports from `first_order.py`, so keep both in one folder) |
| `generated_first_order.txt` | 20 sampled sentences from the first-order model |
| `generated_second_order.txt` | 20 sampled sentences from the second-order model |

## How to run

Requires Python 3.6+. No installs needed.

```bash
python first_order.py          # Parts III-X: CPTs, tests, prediction, generation
python second_order.py         # Parts XI-XIII: second-order model and comparison
python first_order.py 123      # optional: pass an integer seed (default 42)
```

Each run prints its results and (re)writes the generated-sentence text files.

## Model summary

**Tokenisation:** lower-case, split on whitespace, each sentence wrapped as
`<START> w1 ... wn <END>`.

**First order** (network `X1 -> X2 -> ... -> XT`)

```
P(wj | wi) = C(wi, wj) / sum_k C(wi, wk)
```

- Counts: `FirstOrderModel.counts[prev][next]`
- CPT: `FirstOrderModel.probabilities[prev][next]`, built in `_build_probabilities()`

**Second order** (network `X(t-2) -> X(t) <- X(t-1)`)

```
P(c | a, b) = C(a, b, c) / sum_k C(a, b, k)
```

- Counts: `SecondOrderModel.counts[(a, b)][c]`
- CPT: `SecondOrderModel.probabilities[(a, b)][c]`
- Sentences are padded with two `<START>` tokens, so the first word is drawn
  from context `(<START>, <START>)` and the second from `(<START>, w1)`.

**Generation** (both models)

- `generate("sample")`: draw `Xt ~ P(. | context)` until `<END>`
- `generate("greedy")`: always take `arg max`; ties broken alphabetically so
  it is deterministic
- Both stop at `<END>`, at a context never seen in training (no distribution
  exists), or after `MAX_LEN = 20` words. The cap matters: greedy first-order
  decoding loops (`the cat sat on the cat sat on ...`) and would never reach
  `<END>` without it.

**Unseen contexts (Question 7):** `distribution()` returns `None` and
generation stops early rather than crashing. No smoothing is applied, by
design, so zero-probability transitions stay visible.

## Tests

`check_normalisation()` verifies that for every context the probabilities sum
to 1. `first_order.py` also runs it on a deliberately corrupted copy (one
transition deleted), where the total for `the` drops to 0.8333 and the test
fails. That is the behaviour Question 8 asks about.

## Results with the default dataset (seed 42)

| | First-order | Second-order |
|---|---|---|
| Non-zero parameters | 17 | 19 |
| Contexts observed | 11 | 15 |
| Possible contexts (upper bound) | 11 | 121 |
| Full CPT size | 121 | 1331 |
| Distinct sentences in 200 samples | 52 | 6 |
| Samples not in the training data | 46 | 0 |

The first-order model is very diverse but often incoherent (`the park`,
`the cat sat on the cat sat on the mat`). The second-order model is fluent
but only reproduces the six training sentences: more context gives
coherence, but with so little data it amounts to memorisation. This is the
trade-off Question 12 asks about.

## Not included

The written answers to Questions 1-14 and the reflection on how the LLM was
used (Deliverable 6 and 7) are for your lab record. The outputs above give
you the evidence to write them.
