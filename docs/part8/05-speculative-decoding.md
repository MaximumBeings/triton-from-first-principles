# Chapter 27: Speculative Decoding: the Verify Kernel

> Decoding is memory-bound: reading the weights to produce one token costs nearly as much as reading them to check several. Speculative decoding exploits that. A cheap draft model guesses a few tokens ahead, the large target model checks all of them in one pass, and a careful acceptance rule makes the output follow exactly the target's distribution.

!!! info "What was and was not validated"
    The verify kernel and the acceptance rule were run in the **interpreter** (`TRITON_INTERPRET=1`, CPU) on **toy tabular models** (a four-token vocabulary). That validates the algorithm and that the kernel equals a reference. It measures **no speedup**, because a real speedup depends on the draft-to-target cost ratio, batch size and hardware. The 70B multi-GPU measurement belongs to the companion book, [Programming the H100](https://maximumbeings.github.io/programming-the-h100/).

!!! info "Project 7 of the twelve (the kernel half)"
    See [Appendix E](../appendix/inference-projects.md).

## What you will understand by the end of this chapter

- Why verifying several tokens costs about one weight read
- The acceptance rule, and why it is exactly lossless when done correctly
- Why the replacement token after a rejection must come from a **residual** distribution, and what breaks otherwise
- How to compute the expected number of tokens per target pass
- How to write the verify step as a Triton kernel
- What must be true of a real draft and target pair

## What you need to know first

- Chapter 24: decode is memory-bound
- Chapter 26: sampling with Gumbel noise and an argmax
- Chapter 14: reductions (`tl.min`, `tl.max`)

## 27.1 [FOUNDATIONAL] Why It Works

### Intuition

Producing one token with a large model reads all its weights. Checking `K` proposed tokens in a single forward pass reads the same weights, because the `K` positions are processed together as a small batch. So if the draft model proposes `K` tokens and the target accepts most of them, you get several tokens for roughly the price of one target pass.

### Background

```
 draft model (small, fast):   t1 -> t2 -> t3 -> t4         proposes K = 4 tokens
 target model (large):        scores positions 1..5 in ONE pass
 verify:                      accept t1, accept t2, REJECT t3  -> keep t1, t2, plus one new token
 result of this round:        3 tokens for one target pass
```

| Quantity | Meaning |
|---|---|
| `K` | Tokens the draft proposes per round |
| `p` | Target model's distribution at a position |
| `q` | Draft model's distribution at the same position |
| `alpha` | Probability that a proposed token is accepted |

Real speedup depends on `alpha`, `K`, and how much cheaper the draft is than the target. It shrinks at large batch sizes, where the target pass stops being memory-bound.

## 27.2 [FOUNDATIONAL] The Acceptance Rule

### Intuition

The draft samples token `x` from `q`. Accept it with probability `min(1, p(x) / q(x))`. If the target likes the token at least as much as the draft did, always accept. If the target likes it less, accept in proportion. On the first rejection, discard the rest and draw a replacement from the **residual** distribution, the part of `p` that `q` under-covers:

```
residual(v) = max(p(v) - q(v), 0), normalized
```

If all `K` tokens are accepted, draw a bonus token from the target's distribution at position `K+1`; this is the same formula with `q` taken as zero.

### Background

| Case | Action |
|---|---|
| Token accepted | Keep it, check the next |
| First rejection at position `n` | Keep tokens before `n`; replace position `n` with a sample from the residual |
| All `K` accepted | Keep all and add a bonus token sampled from `p` at position `K+1` |
| Every round | Emits between 1 and `K + 1` tokens |

The theorem (Leviathan et al.; Chen et al.) is that this procedure emits tokens distributed **exactly** as if sampled from the target alone. With greedy decoding at temperature zero it reduces to exact-match checking.

!!! warning "COMMON TRAP"
    Resampling the replacement from `p` instead of from the residual looks natural and is wrong: tokens the draft already had a chance to propose get double counted. The result is a biased distribution, and the error is invisible in any single output. Section 27.4 measures it.

## 27.3 [FOUNDATIONAL] The Verify Kernel

### Intuition

One program handles one sequence. It gathers `p(x_i)` and `q(x_i)` for all `K` positions at once, tests `u_i * q(x_i) <= p(x_i)`, which is `u <= p/q` without dividing, finds the first position that fails, then builds the residual at that position and samples it with Gumbel noise.

### Worked Example 27.3.1

```python
@triton.jit
def spec_verify_kernel(p_ptr, q_ptr, x_ptr, u_ptr, g_ptr, n_out_ptr, tok_out_ptr,
                       K, V, KP2: tl.constexpr, VP2: tl.constexpr):
    s = tl.program_id(0)
    i = tl.arange(0, KP2)
    inb = i < K
    x = tl.load(x_ptr + s * K + i, mask=inb, other=0)
    u = tl.load(u_ptr + s * K + i, mask=inb, other=0.0)
    px = tl.load(p_ptr + s * (K + 1) * V + i * V + x, mask=inb, other=1.0)
    qx = tl.load(q_ptr + s * K * V + i * V + x, mask=inb, other=1.0)
    accept = (u * qx) <= px
    n = tl.min(tl.where(inb & (accept == 0), i, K), axis=0)    # K means none rejected
    v = tl.arange(0, VP2); vin = v < V
    pn = tl.load(p_ptr + s * (K + 1) * V + n * V + v, mask=vin, other=0.0)
    qn = tl.load(q_ptr + s * K * V + tl.minimum(n, K - 1) * V + v, mask=vin, other=0.0)
    qn = tl.where(n < K, qn, 0.0)                              # bonus token: residual is p itself
    r = tl.maximum(pn - qn, 0.0)
    g = tl.load(g_ptr + s * V + v, mask=vin, other=0.0)
    score = tl.where(r > 0.0, tl.log(tl.maximum(r, 1e-30)) + g, NEG)
    tl.store(n_out_ptr + s, n)
    tl.store(tok_out_ptr + s, tl.argmax(score, axis=0))
```

The padded lanes (`KP2` and `VP2` round up to powers of two) are given neutral values: `px = qx = 1` and `u = 0`, so a padded position is always accepted and never becomes the first rejection.

### File: `ch27_speculative_decoding.py`

[Download the complete reference](../examples/references/part8/ch27_speculative_decoding.py) (interpreter mode; GPU performance not established).

```bash
TRITON_INTERPRET=1 python3 /Users/oluwaseyiawoga/Developer/triton-from-first-principles/docs/examples/references/part8/ch27_speculative_decoding.py
```

```
kernel == torch reference?  n: True  token: True
```

## 27.4 [FOUNDATIONAL] Proving It Is Lossless, and Checking the Formula

### Intuition

A lossless claim is a statement about distributions, so it needs a statistical test. Use a target model small enough that its exact output distribution can be computed, run speculative decoding thousands of times, and compare.

### Background

The test uses a four-token Markov model for the target and a different, imperfect one for the draft. With three output tokens there are 64 possible sequences. The measure is the total variation (TV) distance between the empirical and the exact distribution: zero means identical, and a perfect sampler still shows a small nonzero value from sampling noise. The noise floor is measured by sampling the target directly.

```
TV distance to exact target distribution (64 sequences, 6000 trials each):
  direct sampling from target (noise floor) = 0.0219
  speculative, correct residual correction  = 0.0193
  speculative, WRONG correction (from p)    = 0.2093
```

The correct procedure is indistinguishable from the noise floor. The wrong correction is about ten times further from the target, even though every individual output looks plausible.

### Worked Example 27.4.1: tokens per target pass

If each proposed token is accepted independently with probability `alpha`, the expected number of tokens emitted per target pass is

```
E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)
```

For a pair with per-token acceptance `alpha = sum over v of min(p(v), q(v))`:

```
alpha = 0.391, K = 4: tokens per target pass  measured = 1.630  formula (1-a^(K+1))/(1-a) = 1.628
```

The speedup is this number divided by the relative cost of a round: one target pass plus `K` draft passes. A cheap draft with a high `alpha` wins; a draft that is slow or often wrong does not.

!!! warning "UNTESTED ON GPU"
    No speedup was measured. This chapter proves the sampler correct; it does not show it is fast.

## 27.5 Requirements for a Real Pair

| Requirement | Why |
|---|---|
| Same tokenizer and vocabulary | `p(x)` and `q(x)` must refer to the same token ids |
| Rejection sampling at temperature above zero | Greedy matching is only valid at temperature zero |
| KV-cache rollback | Rejected positions must be removed from both models' caches |
| Draft not too slow | Draft cost counts against the saving |
| Measured acceptance by task | Code and extraction usually accept more than open chat |
| Smaller gain at large batch | Verification stops being nearly free once the target pass is compute-bound |

Related methods change where the draft comes from: n-gram or prompt lookup, extra prediction heads (Medusa), or a feature-level draft (EAGLE). The verify step is the same.

## 27.6 [FOUNDATIONAL] Self-Tests: Proving the Kernel and the Algorithm

`main()` shows the kernel agreeing with a reference and the output distribution matching the target. The `--test` flag turns that into 20 checks that fail when something is wrong:

```bash
TRITON_INTERPRET=1 python3 ch27_speculative_decoding.py --test
```

```text
[PASS] kernel == torch reference for (K, V) = (1,2) (2,3) (3,6) (4,7) (5,16) (8,50), most of them not powers of two
  [PASS] the number accepted is always between 0 and K
  [PASS] the extra token always has positive residual mass: p - q > 0 after a rejection, p_K > 0 after a full accept
  [PASS] a one-token vocabulary (V = 1) works: everything accepted, token 0
  [PASS] hand case, boundary: p(x)=0.25, q(x)=0.5; u=0.5 gives u*q = p exactly and is ACCEPTED, u=0.5001 is rejected (kernel and torch agree)
  [PASS] hand case, first rejection decides: accept pattern [yes yes NO yes] gives n = 2, the 4th token is discarded even though it would have been accepted
  [PASS] p == q exactly: every one of 200 rows accepts all 3 draft tokens, and the bonus token comes from p_K (here a one-hot on token 1)
  [PASS] the target gives the draft token probability 0: every row is rejected (n = 0), token 0 is never returned, and tokens 1 and 2 appear with frequency 0.3 and 0.7 within 0.04
  [PASS] documented edge: u = 0 accepts a draft token the target gives probability 0 (0 * q <= 0); a measure-zero event for u drawn from [0, 1)
  [PASS] documented edge: a draft token with q = 0 (impossible for a real draft model to sample) is accepted without dividing by zero
  [PASS] documented limitation: if the residual is empty (p <= q everywhere, which only happens for unnormalized p) the kernel and torch both return token 0, which p gives probability 0
  [PASS] documented limitation: float16 probabilities are rejected (tl.log wants float32); pass float32 p and q
  [PASS] exact identity on 200 random (p, q): P(accept x) + P(reject) x normalized residual equals p(x) to 1e-12 (worst error 3.3e-16)
  [PASS] the same algebra with the WRONG correction (resample from p) is biased: mean TV to p is 0.135 over the same 200 pairs
  [PASS] through the REAL kernel, K = 1: the first emitted token follows p = (.1 .2 .3 .4) within 0.035 over 4000 rows although the draft is q = (.4 .3 .2 .1); max deviation 0.004
  [PASS] 3-token sequences from a Markov-chain target (64 outcomes, 6000 trials): correct correction TV 0.027 < 0.04, wrong correction TV 0.236 > 0.10
  [PASS] tokens per target pass match (1 - a^(K+1)) / (1 - a) within 0.03 for K = 1, 2, 4, 8
  [PASS] disjoint supports (acceptance rate 0): every row is rejected at position 0, so a step yields exactly 1 token
  [PASS] the kernel is deterministic and does not modify its inputs
  [PASS] rows are independent: rewriting row 0's target distribution leaves rows 1..15 unchanged
20/20 self-tests pass
```

It takes about 50 seconds.

Speculative decoding is unusual because its correctness is a mathematical claim: the output must have exactly the target's distribution. The suite checks that claim four different ways, because no single one is enough.

1. **Algebra (test 13).** For 200 random pairs `(p, q)` it computes, in `float64`, the probability that token `x` is emitted: the chance the draft proposes `x` and it is accepted, plus the chance of any rejection times the normalized residual at `x`. That equals `p(x)` to 3e-16. This is the proof, checked numerically, with no sampling noise.
2. **The real kernel, one step (test 15).** With `K = 1`, a draft `q = (.4 .3 .2 .1)` and a target `p = (.1 .2 .3 .4)` that disagree strongly, the first token emitted by the kernel over 4000 rows is within 0.004 of `p`. A tolerance of 0.035 is about 4.4 standard deviations of a binomial proportion at that size, so a correct kernel essentially never fails it.
3. **Multi-step sequences (test 16).** A three-token Markov-chain target with 64 possible outputs: the correct procedure has total-variation distance 0.027 from the exact distribution, the wrong correction 0.236.
4. **Hand cases (tests 5 to 8).** The accept boundary, the first-rejection rule, the all-accepted bonus token and the all-rejected residual, each with an answer worked out on paper.

**Do the tests catch bugs?** Each row is one change to a copy of the file. The number is how many of the 20 tests then fail.

| Broken copy | Tests failing | Verdict |
|---|---|---|
| accept test `<` instead of `<=` | 2 | caught (hand boundary test) |
| accept test `u * px <= qx` (ratio inverted) | 8 | caught |
| last rejection used instead of the first | 8 | caught |
| bonus token not stripped of `q` | 1 | caught (test 7 only) |
| replacement drawn from `p`, not the residual | 4 | caught |
| noise dropped from the replacement sample | 3 | caught |
| `r + g` instead of `log r + g` | 4 | caught |
| `p` batch stride `K` instead of `K + 1` | 4 | caught |
| residual guard `r > 0` removed | 1 | caught (test 11 only) |
| torch reference rejects at `>=` | 1 | caught (boundary test) |
| Markov chain forgets the previous token | 1 | caught |
| torch reference ignores the wrong-correction flag | 1 | caught: the "wrong" run becomes right, so test 16's bound fails |
| replacement token `argmin` instead of `argmax` | crash | caught |
| residual row index `n` unclamped (`nq = n`) | 0 | **survives** |
| `px` padding value 0 instead of 1 | 0 | **survives** |
| `inb` mask removed from the first-rejection reduction | 0 | **survives** |
| padded lanes forced to reject, and no `inb` mask | 0 | **survives** |

**Why a statistical test cannot catch the `<` mutant.** The boundary `u * q == p` has probability zero for a continuous `u`, so no amount of sampling will ever see `<` versus `<=`. Only the hand case, which sets `u * q` exactly equal to `p`, does. That is a general point: tests for exact semantics at a boundary must be constructed, not sampled.

**The four survivors are not test gaps.**

- `nq = n`: when everything is accepted, `n = K` would read a row of `q` past the end, but the value is zeroed immediately afterwards by `n < K`. The mask protects an address, and the interpreter does not fault on it.
- The padding rows: rounding `K` up to a power of two adds padded lanes, and the chapter's explanation was that they get `u = 0` and `px = qx = 1` so they can never be the first rejection. The mutants show that is only one of three protections. The `inb` mask excludes them, and even a padded lane that *is* rejected gives `n = K`, because the first padded index is exactly `K`. Removing the values, the mask, or both together changes no output. The values are harmless to keep, but they are not what makes the kernel correct.

## 27.7 [FOUNDATIONAL] Experiments

Five experiments, run with `--extras` (about 2 seconds).

```bash
TRITON_INTERPRET=1 python3 ch27_speculative_decoding.py --extras
```

```text
Experiment A: draft quality against acceptance (q = lam p + (1 - lam) r, V = 8, K = 4, 20000 trials)
lam    alpha   tokens/pass measured   formula
1.00   1.000               5.000      5.000
0.75   0.855               3.732      3.745
0.50   0.710               2.836      2.826
0.25   0.565               2.166      2.166
0.00   0.420               1.710      1.702

Experiment B: a MODEL of speedup = tokens per pass / (1 + K c), c = draft cost / target cost (assumes one verify pass costs the same for any K: optimistic)
alpha    c     best K   tokens/pass   modelled speedup
 0.50  0.05       3          1.88         1.63x
 0.50  0.20       1          1.50         1.25x
 0.70  0.05       6          3.06         2.35x
 0.70  0.20       3          2.53         1.58x
 0.80  0.05       8          4.33         3.09x
 0.80  0.20       4          3.36         1.87x
 0.90  0.05      12          7.46         4.66x
 0.90  0.20       7          5.70         2.37x
(K is searched from 1 to 12; alpha = 0.9 with c = 0.05 hits that cap.) A draft costing a fifth of the target with alpha = 0.5 gives a modest 1.25x in this model. None of this is a measurement.

Experiment C: how biased is the wrong correction? (random 8-token pairs; TV distance of the first emitted token from p)
alpha   TV(correct)   TV(resample from p)
0.349       0.0e+00            0.155
0.412       0.0e+00            0.212
0.384       9.0e-17            0.130
0.343       6.9e-18            0.211
0.495       4.5e-17            0.121
0.144       8.7e-19            0.093
the wrong correction over-represents every token with q >= p (their probability becomes p(1 + 1 - alpha)); the bias shrinks only as alpha approaches 1

Experiment D: why the kernel tests u * q <= p and does not divide
float32 p/q with q = 1e-45: inf   with p = q = 0: nan   (inf and nan)
float32 u*q <= p with the same values and u = 0.5: True, True   (finite comparisons, no special cases)

Experiment E: bytes the verify kernel reads per sequence (float32 p and q; it reads K accept scalars plus ONE row of p and ONE row of q)
V        K    bytes read   full p, q tensors   share
  32000   4      256032             1152000   22.2 %
  32000   8      256064             2176000   11.8 %
 128256   4     1026080             4617216   22.2 %
 128256   8     1026112             8721408   11.8 %
the target model produces the full tensors regardless; the kernel avoids re-reading most of them
```

**A. Draft quality.** The measured tokens per pass track `(1 - a^(K+1)) / (1 - a)` to within 0.013 across the range from a perfect draft (5.000, all `K + 1` tokens) to an independently drawn draft (1.71 at `alpha = 0.42`). Even an unrelated draft never yields fewer than one token per pass, because the replacement token is always produced.

**B. A model of speedup.** Dividing tokens per pass by `1 + K c` (with `c` the draft's cost as a fraction of the target's) gives the pattern in the table: the best `K` rises with `alpha` and falls as the draft gets more expensive. This is a model, and an optimistic one: it assumes the verify pass costs the same for any `K`, which stops being true once the target pass is compute-bound at large batch. The search covers `K` from 1 to 12, and `alpha = 0.9` with `c = 0.05` reaches that cap. No speed was measured.

**C. Bias of the wrong correction.** Resampling from `p` instead of the residual gives a first-token distribution 0.09 to 0.21 away from `p` in total variation for random pairs with `alpha` between 0.14 and 0.50. It over-represents every token with `q >= p`, whose probability becomes `p(2 - alpha)`. The correct rule is exact to rounding error (1e-17).

**D. Why no division.** In `float32`, `p / q` with `q = 1e-45` is `inf` and with `p = q = 0` is `nan`. The kernel's `u * q <= p` stays finite for both.

**E. Bytes read.** The kernel reads `2K` scalars and one row each of `p` and `q`, which is 22% of the full tensors at `K = 4` and 12% at `K = 8`, for either a 32,000 or a 128,256-token vocabulary. The target model has already produced the full tensors, so this is what the kernel adds, not a speed claim.

## 27.8 [FOUNDATIONAL] Failure Modes and Edge Cases

| Situation | What happens | Test |
|---|---|---|
| `u * q == p` exactly | Accepted (the rule is `<=`) | 5 |
| First rejection is not the last | Tokens after it are discarded | 6 |
| `p == q` | Everything accepted; bonus token from `p_K` | 7 |
| Target gives the draft token probability 0 | Always rejected; replacement drawn only from `p`'s support | 8 |
| `K` or `V` not a power of two; `V = 1` | Correct | 1, 4 |
| `u = 0` | Accepts even a token with `p = 0` (probability zero for random `u`) | 9 |
| Draft token with `q = 0` | Accepted; no division | 10 |
| Residual empty (`p <= q` everywhere, i.e. unnormalized `p`) | Returns token 0, which `p` gives probability 0 | 11 |
| `float16` probabilities | Rejected: `tl.log` needs `float32` | 12 |
| Acceptance rate 0 | One token per pass, still correct | 18 |
| Draft token id outside `[0, V)` | Not checked | not modelled |

**Not modelled.** Any real model pair: tokenizer agreement, the target's actual `p` at temperature, top-k and top-p filtering applied to `p` and `q` (which must be applied consistently to both for the distribution guarantee to hold); KV-cache rollback of rejected positions; the draft's own cost; batching rows that accepted different numbers of tokens; tree-structured drafts; sampling the draft with different noise than the verify; `float16` and `bfloat16` probabilities; and all timing. The Markov-chain target has 4 tokens, so the agreement says the algorithm is exact, not that any real pair gains speed.

## 27.9 [FOUNDATIONAL] On Real Hardware (UNTESTED)

Everything above ran on CPU. These commands were written without access to a GPU and have **not been run by the author**; verify them against your installed versions.

```bash
# 1. Correctness on the GPU (no TRITON_INTERPRET)
python3 ch27_speculative_decoding.py --test

# 2. Verify-kernel time at two vocabulary sizes and three batch sizes
python3 ch27_speculative_decoding.py --bench
```

**Known gaps before `--bench` means anything:** `verify_kernel` allocates its outputs on the CPU, so it needs `device=p.device`; and it launches one program per sequence that loads a row of `next_power_of_2(V)` elements, which for a 128,256-token vocabulary is 131,072 elements and may not fit in one program's registers. First check that it launches, then rewrite the residual step to loop over the vocabulary in tiles. A reasonable hypothesis, not a result: the kernel is small next to the target's forward pass, so the speedup of a real system depends on the acceptance rate and the draft's cost far more than on this kernel. To test that, measure acceptance on your own task and plug it into the model in experiment B.

## Chapter Summary

Speculative decoding has a cheap draft propose `K` tokens and the target verify them in one pass, which costs about one weight read. Each token is accepted with probability `min(1, p/q)`, and on the first rejection the replacement is drawn from the normalized residual `max(p - q, 0)`; with all `K` accepted, a bonus token is drawn from `p`. A Triton verify kernel matched a reference exactly. A statistical test against an exactly known target distribution showed the correct procedure at the sampling noise floor (0.0193 against 0.0219) and the wrong replacement rule far off (0.2093). The tokens-per-pass formula matched simulation (1.630 against 1.628). Speedup itself was not measured and depends on the draft cost, acceptance and batch size.

## Self-Check Questions

1. Why does verifying `K` tokens cost about the same as generating one?
2. Why must the replacement token come from the residual `max(p - q, 0)` and not from `p`?
3. With `alpha = 0.8` and `K = 4`, what is the expected number of tokens per target pass?
4. Why does the kernel give padded positions `u = 0` and `px = qx = 1`?
5. Name three reasons real-world speedup could be lower than the formula suggests.

## Where We Go Next

This is the last chapter of Part 8. The companion book, [Programming the H100](https://maximumbeings.github.io/programming-the-h100/), picks up the systems side: KV-cache offloading, disaggregated serving, mixture-of-experts, and running a 70B target across several GPUs.

## Worked Solutions

1. Decode is limited by reading the weights from memory, and the `K` positions are processed together as one small batch, so the same weight read serves all of them.
2. The draft has already had its chance to propose each token. Resampling from `p` counts that probability mass twice, which biases the output. The residual `max(p - q, 0)` is exactly the mass `q` failed to cover, and the measured TV distance (0.2093 against 0.0193) shows the bias.
3. `(1 - 0.8^5) / (1 - 0.8) = (1 - 0.32768) / 0.2 = 3.3616`, about 3.36 tokens per target pass.
4. With `u = 0`, `0 * qx <= px` always holds, so a padded position is accepted and cannot be the first rejection, and `px = qx = 1` keeps the test well defined. These values are one of three independent protections, not the one that matters: the `inb` mask also excludes padded lanes, and even a padded lane that were rejected would give `n = K` because the first padded index is exactly `K`. The mutation results in section 27.6 show that removing the values, the mask, or both changes no output.
5. The draft model's own cost is not free; acceptance is below the independence assumption on real text and varies by task; large batches make the target pass compute-bound; and rolling back rejected KV entries adds overhead.

## Additional Exercises

6. In experiment B, `alpha = 0.8` with `c = 0.2` has its best at `K = 4`, but `c = 0.05` pushes it to 8. Explain why, and say why the model overstates the gain at large batch.
7. Prove that padded lanes cannot change `n`, whichever values they hold, using only the fact that the first padded index equals `K`.
8. When the residual is empty the kernel returns token 0. Where would you handle it, and what is a safe replacement?
9. Justify the 0.035 tolerance in test 15 and say how far the wrong correction is from `p` for that example, in standard deviations.
10. Explain why a sampling test cannot distinguish `<` from `<=` in the accept rule, and what test does.

## Additional Solutions

6. A longer draft block is worth its extra draft cost only if the extra tokens are likely to be accepted. A higher `alpha` makes each extra token more likely to survive, and a smaller `c` makes each one cheaper, so both push the best `K` up. At large batch the target pass becomes compute-bound and its cost grows with the `K + 1` positions it verifies, which the model assumes it does not, so the real best `K` and the real gain are smaller.
7. `n` is the minimum over lanes of (lane index if rejected, else `K`). A padded lane has an index of at least `K`, so whether it is rejected (index at least `K`) or accepted (value `K`), it contributes a value of at least `K`, and `n` is already at most `K` from the default. It can therefore never lower `n`, and the real lanes alone decide.
8. In the target-side code that prepares `p` and `q`, because the kernel cannot raise an error. If `(p_n - q_n).clamp(0)` sums to zero, fall back to sampling from `p_n`, which is the right answer when the residual vanishes numerically (the two distributions are equal up to rounding, so a rejection there was a rounding artifact).
9. The first-token frequency over 4000 rows has a standard deviation of at most `0.5 / sqrt(4000) = 0.0079`, so 0.035 is about 4.4 standard deviations. Under the wrong correction the probabilities become `.14, .28, .32, .26` against `p = .1, .2, .3, .4`, so token 3 is off by 0.14, which is about 18 standard deviations. The test separates the two cleanly.
10. The boundary `u * q == p` has probability zero for continuous `u`, so no sample ever lands on it and `<` and `<=` give the same distribution. A hand case that sets `u * q` exactly equal to `p`, like test 5, is the only test that sees the difference.

## Files and Sources

- `examples/references/part8/ch27_speculative_decoding.py`: verify kernel, torch reference, Markov-chain experiment, `main()`, `--test` (20 self-tests, about 50 s), `--extras` (experiments A to E), `--bench` (GPU only, untested, with the gaps noted above).
- Outputs shown were produced with `TRITON_INTERPRET=1` on CPU, Triton 3.9.0 and torch 2.15.
- The speedup model in experiment B is arithmetic on assumed costs, not a measurement. No real model pair was run.
