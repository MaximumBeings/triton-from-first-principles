# Chapter 26: Grammar-Constrained Decoding

> A language model emits probabilities over tokens. A grammar says which tokens may legally come next. Put a mask between them, and the output cannot be malformed. Almost all of the work is in building that mask cheaply; the GPU kernel that applies it is small.

!!! info "What was and was not validated"
    The grammar automaton, the mask compiler, the cache and the Triton sampling kernel were run in the **interpreter** (`TRITON_INTERPRET=1`, CPU) with a **toy 24-token vocabulary and random logits**. That validates the mechanism: masks are correct, the kernel matches a reference, and constrained output is valid. It says nothing about real-model output quality, a real tokenizer's cost, or GPU timing. The overlap strategy in Section 26.4 is a design, not a measurement.

!!! info "Project 4 of the twelve"
    See [Appendix E](../appendix/inference-projects.md).

## What you will understand by the end of this chapter

- Why constrained decoding is about 80 percent host-side work and 20 percent kernel
- How a pushdown automaton tracks "where we are" in a JSON grammar, and why a stack is needed
- How to compile a mask per automaton state, cache it, and pack it into bits
- Why tokens that span several characters make the mask harder than it looks
- How a Triton kernel applies a bitmask and samples in one pass
- Why "100 percent valid" needs qualifying, and what the qualifications are

## What you need to know first

- Chapter 14: reductions (the sampling step is an argmax reduction)
- Chapter 21: softmax and sampling-adjacent kernels
- Chapter 23: single-program-per-row kernels

## 26.1 [FOUNDATIONAL] The Mask Is the Whole Idea

### Intuition

At each step the model produces one logit per vocabulary entry. A grammar defines, given the text so far, which next tokens keep the output on a path that can still become valid. Set every other logit to minus infinity before sampling, and the model can only choose legal tokens. The model still ranks the legal ones; the grammar only removes the illegal ones.

### Background

```
 text so far:  {"ab":                 grammar state: just read a key, now expecting a VALUE
 vocab:        { } [ ] , : " ab cd xyz 0 1 23 7 true false null ... <eos>
 allowed:      {   [   "          0 1 23 7 true false null          (a value can start)
 masked out:     }   ] , :  ab cd xyz                    <eos>      (illegal here)
```

| Piece | Where it runs | Cost |
|---|---|---|
| Track the grammar state | Host (CPU) | Per token, per sequence |
| Compute the mask for that state | Host (CPU), then cached | Expensive the first time, free after |
| Apply the mask and sample | GPU kernel | Tiny |

The expensive part is deciding, for a vocabulary of 100,000 or more tokens, which are legal in the current state. A naive version checks every token at every step.

## 26.2 [FOUNDATIONAL] A Pushdown Automaton for JSON

### Intuition

JSON nests: an object can contain an array that contains an object. A finite automaton cannot count how deeply it is nested, so it cannot know whether `}` or `]` closes the current container. A pushdown automaton adds a stack of open containers. This chapter's grammar is a JSON subset: objects, arrays, lowercase strings, non-negative integers, `true`, `false` and `null`, no whitespace, nesting depth at most 3.

### Background

The state is a triple: a mode (what we expect next), the stack of open containers, and, for literals, the characters still owed.

| Mode | Meaning | Legal next characters |
|---|---|---|
| `VALUE` | A value must start | `{ [ " 0-9 t f n` |
| `OBJK` | Just after `{` | `"` (key) or `}` |
| `COLON` | After a key | `:` |
| `AFTER` | After a complete value | `,` or the closer for the top of the stack; nothing if the stack is empty |
| `STR`, `KEYSTR` | Inside a string | lowercase letters or closing `"` |
| `NUM`, `NUM0` | Inside an integer | digits (none after a leading `0`) |
| `LIT` | Inside `true`, `false`, `null` | the next expected letter |

!!! warning "COMMON TRAP"
    JSON forbids leading zeros: `01` is invalid. A grammar that lets digits follow a leading `0` produces outputs that look right and fail `json.loads`. The `NUM0` mode exists to prevent exactly that.

## 26.3 [FOUNDATIONAL] Tokens Are Not Characters

### Intuition

A model's vocabulary contains multi-character tokens, and several of them straddle grammar boundaries. The token `":"` closes a key string, reads a colon, and opens a value, all at once. Checking a token means running **every one of its characters** through the automaton from the current state; the token is legal only if all of them are.

### Background

```
 state: inside a key string      token '":"'
   '"' -> closes the key        -> COLON
   ':' -> reads the colon       -> VALUE
   '"' -> opens a string value  -> STR          all steps legal => token allowed
```

The mask for a state is the set of tokens whose characters all step cleanly. Because the same state recurs constantly (inside a string, after a value, and so on), the mask is computed once and cached by state, then stored as packed bits: bit `i` of word `i // 32` says whether token `i` is allowed.

### Worked Example 26.3.1: the mask cache

Across 300 simulated generations the compiler built masks for only 68 distinct automaton states and served the rest from cache:

```
mask cache: 68 distinct states compiled, hits=23934, misses=68
```

On a real tokenizer the number of distinct states is larger, and caching strategies are an active area of engineering; see the XGrammar and Outlines projects for production approaches.

## 26.4 The Sampling Kernel

### Intuition

The kernel takes the logits, the packed mask and Gumbel noise supplied by the host, sets masked scores so low they can never win, and returns the argmax. Adding Gumbel noise to `logit / temperature` and taking the argmax is an exact way to sample from the softmax distribution without computing the softmax.

### Worked Example 26.4.1

```python
@triton.jit
def masked_sample_kernel(logits_ptr, mask_ptr, noise_ptr, out_ptr, V, n_words, inv_temp,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    inb = i < V
    word = tl.load(mask_ptr + row * n_words + i // 32, mask=inb, other=0)
    allowed = ((word >> (i % 32)) & 1) == 1
    x = tl.load(logits_ptr + row * V + i, mask=inb, other=0.0).to(tl.float32) * inv_temp
    g = tl.load(noise_ptr + row * V + i, mask=inb, other=0.0)
    score = tl.where(allowed & inb, x + g, -1.0e30)     # disallowed tokens can never win
    tl.store(out_ptr + row, tl.argmax(score, axis=0))
```

### File: `ch26_grammar_decoding.py`

[Download the complete reference](../examples/references/part8/ch26_grammar_decoding.py) (interpreter mode; GPU performance not established).

```bash
TRITON_INTERPRET=1 python3 /Users/oluwaseyiawoga/Developer/triton-from-first-principles/docs/examples/references/part8/ch26_grammar_decoding.py
```

```
kernel == reference? True
constrained: finished 298/300; valid among finished = 298/298; truncated at max_tokens = 2
constrained, ALL outputs (incl. truncated) valid: 298/300
unconstrained: valid JSON = 4/300
mask cache: 68 distinct states compiled, hits=23934, misses=68
sample constrained outputs: ['true', '[{},"",{},{"cdtrue":""},{},[0],777017,null]', '[{}]']
```

Read the second and third lines carefully. Every output that **finished** was valid, 298 of 298. But two of the 300 hit the 80-token limit mid-structure, and a truncated JSON document is not valid. So "valid among finished" is 100 percent while "valid overall" is 298 of 300.

!!! warning "COMMON TRAP"
    "The grammar guarantees valid JSON" is true only for outputs that **reach the end of the grammar**. A `max_tokens` cutoff, a stop condition, or a timeout can end generation mid-structure. Real systems either reserve budget to close open containers, force closing tokens near the limit, or treat truncation as a failure and report it.

!!! warning "UNTESTED ON GPU"
    Overlapping host mask computation with the GPU forward pass is the main performance lever: while the GPU runs step `t`, the CPU computes masks for step `t+1`'s likely states, so the mask is ready when the logits are. This is a design, not a measurement here.

## 26.5 What the Guarantee Does Not Cover

| Limitation | Why it matters |
|---|---|
| Truncation | Outputs cut off by `max_tokens` can be invalid, as measured above |
| End of sequence | The end token must be allowed only when the grammar is complete (this chapter gates it with `can_end`) |
| Schema constraints | Pydantic features such as regex patterns, numeric bounds and uniqueness do not map cleanly onto a context-free grammar; many are enforced by validation afterwards |
| Quality | Forcing tokens changes the distribution; the model may prefer a different, illegal token, and the legal choice can be worse |
| Cost | Compiling masks for a large grammar and vocabulary can dominate if not cached |
| Plausible but wrong | Syntactically valid output can still be semantically wrong; the grammar guarantees form, not truth |

A fair evaluation compares constrained and unconstrained generation on a schema-conformance benchmark, reports the truncation rate separately, and measures end-task accuracy, not just validity.

## 26.6 [FOUNDATIONAL] Self-Tests: Proving the Automaton, the Masks and the Kernel

`main()` shows constrained outputs parsing as JSON. That is a sample, not a proof. The `--test` flag runs 29 checks, most of them about the automaton, because the mask is only as good as the grammar behind it:

```bash
TRITON_INTERPRET=1 python3 ch26_grammar_decoding.py --test
```

```text
[PASS] exploring every token from every state reaches 110 states (a regression value: it changes if the grammar or vocabulary does)
  [PASS] no dead state: every reachable state allows at least one token (counting <eos>)
  [PASS] no trap state: from every reachable state some token sequence reaches a state where <eos> is allowed
  [PASS] completeness: 400 random JSON values in the supported subset (depth <= 3), written compactly, are all accepted character by character
  [PASS] soundness: 3000 random walks through allowed tokens that end at <eos> are all valid JSON (finished: 2963)
  [PASS] 17 malformed strings (leading zero, trailing comma, unclosed, depth 4, ...) are rejected
  [PASS] 12 well-formed strings (including depth 3, empty containers and the empty key) are accepted
  [PASS] documented subset: 9 strings that real JSON accepts are rejected here (negative, decimal, exponent, uppercase, space, escape, digit in key)
  [PASS] tokenization independence: two different token splits of the same text reach the identical final state
  [PASS] for all 110 states, mask bit i equals 'every character of token i is accepted' (recomputed independently with step), and the <eos> bit equals can_end
  [PASS] hand case: at the start the allowed tokens are exactly { [ " 0 1 23 7 true false null {" [{ (12 tokens)
  [PASS] hand case: after {"ab":1 only more digits, a comma or a closing brace are allowed, not <eos> and not a quote
  [PASS] hand case: after a top-level 1, <eos> is allowed because a number may end there, and ',' is not
  [PASS] cache: the first lookup is a miss, the second returns the same tensor and counts a hit
  [PASS] a 71-token vocabulary packs into 3 int32 words, with a negative word when bit 31 is set, and unpacks exactly
  [PASS] kernel == torch reference on 3 batches of 48 random masks at temperature 1, 0.5 and 2
  [PASS] the kernel never returns a token whose mask bit is clear (when at least one bit is set)
  [PASS] V = 70 (3 words, BLOCK 128): a mask with only token r allowed returns r for every r, across the bit-31/32/63/64 word boundaries
  [PASS] V = 32 (exactly one word) and V = 33 (spills into a second word) match the reference
  [PASS] temperature 1e-3 is greedy: the kernel returns the highest-logit ALLOWED token
  [PASS] with zero noise the kernel is a masked argmax of the logits
  [PASS] documented limitation: a row with NO allowed token silently returns token 0, which is disallowed (the caller must prevent empty masks)
  [PASS] Gumbel-max sampling follows softmax over the allowed tokens: 200,000 draws within 0.01 of the exact probabilities
  [PASS] the kernel is deterministic and does not modify its inputs
  [PASS] constrained decode, 40 rows x 50 tokens: every FINISHED output is valid JSON (finished: 40/40)
  [PASS] every output, including the ones cut off at max_tokens, is a live prefix: the automaton is still alive and could still finish
  [PASS] the mask cache holds only states that are reachable (at most 110)
  [PASS] unconstrained decode of the same shape: fewer than half the outputs are valid JSON
  [PASS] the same seed reproduces the outputs exactly, and a different seed changes them
29/29 self-tests pass
```

The suite takes about 15 seconds, against about two minutes for `main()`, because it never runs the 300-row decode.

Three kinds of check carry most of the weight.

**Exhaustive structure.** The automaton's state includes the whole stack, but the stack depth is capped, so the set of reachable states is finite. Test 1 explores every token from every state and finds 110 of them. Test 2 shows that none is a dead end, and test 3 shows that from every state some token sequence reaches a point where `<eos>` is allowed. Together they prove that a constrained decode can never get stuck with no legal token, which a sample of 300 outputs could not.

**Differential tests against the real parser.** Test 4 generates 400 random JSON values in the supported subset, writes them compactly with `json.dumps` and requires the automaton to accept every one. Test 5 runs 3000 random walks through allowed tokens and requires `json.loads` to accept every finished one. Between them the automaton is shown to be neither too strict nor too loose for the subset it claims. Tests 6 to 8 add the boundary: 17 malformed strings rejected, 12 well-formed ones accepted, and 9 strings that real JSON allows but this grammar deliberately does not (negative numbers, decimals, exponents, uppercase, spaces, escapes, digits in keys).

**Hand-derived masks.** Tests 11 to 13 give the allowed token set at three states, worked out on paper from the grammar and not read from the code. At the start the 12 legal tokens are `{ [ " 0 1 23 7 true false null {" [{`. After `{"ab":1` the legal tokens are more digits, a comma or `}`. After a top-level `1`, `<eos>` is legal, because a number may end there.

**Do the tests catch bugs?** Each row is one change to a copy of the file. The number is how many of the 29 tests then fail.

| Broken copy | Tests failing | Verdict |
|---|---|---|
| empty arrays impossible | 2 | caught |
| leading zero allows more digits (`01`) | 3 | caught |
| nesting limit 4 instead of 3 | 2 | caught |
| trailing comma allowed in objects | 3 | caught |
| trailing comma allowed in arrays | 3 | caught |
| a number cannot end the document | 5 | caught |
| literals (`true`) never complete | 3 | caught |
| `run_token` resets instead of failing | 7 | caught |
| `<eos>` always allowed | 4 | caught |
| digits allowed in keys and strings | 1 | caught, by the "documented subset" test only: it is a design decision, not a bug |
| signed-int32 conversion removed | crash | caught: bit 31 overflows `int32` |
| kernel word index `i // 16` | 7 | caught |
| kernel bit index `i % 31` | 2 | caught |
| kernel ignores the temperature | 2 | caught |
| kernel noise dropped | 2 | caught |
| disallowed score `0` instead of `-1e30` | 2 | caught |
| `argmin` instead of `argmax` | 7 | caught |
| decode loop never advances the automaton | 2 | caught |
| `& inb` removed from the score | 0 | **survives** |
| mask-word load unmasked | 0 | **survives** |
| both of the last two removed together | 1 | caught, by the V = 70 test only |

Several of the kernel rows end in a crash and not a clean failure: a wrong kernel picks a disallowed token, the decode loop feeds it to the automaton, the automaton returns `None` and the next mask lookup raises. The suite records that as a failed check, so the number above counts it.

**The last three rows teach something.** Lanes past `V` are protected twice: the mask-word load returns 0 for them, and the score is forced low by `& inb`. Remove either one and nothing changes, because the other still covers. Remove both and a padding lane scores 0, which can beat the real tokens' negative scores, so the kernel can return a token index past the vocabulary. Only the V = 70 test notices, which is why it stays: it is the only line of defence once someone "simplifies" one of the two. This is the general lesson of mutation testing on defensive code: two redundant guards each look removable, and the suite must test the case where both are gone.

## 26.7 [FOUNDATIONAL] Experiments

Five experiments, run with `--extras` (about 45 seconds). They use the toy vocabulary.

```bash
TRITON_INTERPRET=1 python3 ch26_grammar_decoding.py --extras
```

```text
Experiment A: size of the state space (toy vocabulary, 24 tokens, MAXDEPTH 3)
reachable automaton states: 110; distinct masks: 19; bytes per mask: 4
allowed tokens per state: min 1, median 6, mean 5.28, max 13 of 24; states with exactly one allowed token: 16

Experiment B: how the state count grows with the nesting limit (same vocabulary)
MAXDEPTH  states
       1      20
       2      50
       3     110
       4     230
       5     470
       6     950
the stack is part of the state, so the count roughly doubles per level: a cache keyed by full state does not stay small for deep grammars

Experiment C: validity against the output budget (100 rows per cell)
max_tokens  constrained: finished  valid-of-finished  cut off   unconstrained: valid
        10                   78          78/78         22                   1
        30                   93          93/93          7                   1
        60                  100         100/100         0                   1
a cut-off output is a valid PREFIX, not valid JSON: the guarantee needs enough budget to reach the end

Experiment D: tokenization ambiguity (number of token sequences that spell the same text)
{"ab":1}           vocabulary only:   2   accepted by the grammar:   2
{"ab":"cd"}        vocabulary only:   8   accepted by the grammar:   8
[{"xyz":[23]}]     vocabulary only:   3   accepted by the grammar:   3
the mask must therefore be a function of the automaton state alone, never of how the text was split into tokens

Experiment E: mask size for a large vocabulary (assumed 128,256 tokens, like a Llama-3 tokenizer)
4008 int32 words = 16032 bytes (15.7 KiB) per mask; 19 distinct masks would need 297 KiB, all 110 states 1.68 MiB
compiling one mask by this method runs the automaton over every token: 128256 tokens x (average token length, say 6) = 769,536 character steps in Python
```

**A. State space.** 110 states but only 19 distinct masks, at 4 bytes each. Each state allows between 1 and 13 of the 24 tokens, 5.3 on average, and 16 states allow exactly one. The decode in `main()` visited 68 of the 110, so a cache is filled lazily and never completes.

**B. State growth.** The count roughly doubles with every extra level of nesting: 20, 50, 110, 230, 470, 950 for depth 1 to 6. The stack is part of the state, so a cache keyed by the full state grows exponentially with the depth limit, while the number of distinct masks stays small. Exercise 6 asks for a smaller cache key.

**C. Validity against the output budget.** With a 10-token budget, 78 of 100 constrained rows finished and every one parsed; 22 were cut off. At 30 tokens 7 were cut off, and at 60 none. Unconstrained output was valid for 1 row in 100 at every budget. The guarantee is about the grammar, not about the budget: a cut-off output is a valid prefix, not valid JSON. The truncation rate must be reported next to the validity rate.

**D. Tokenization ambiguity.** The text `{"ab":"cd"}` can be spelled with 8 different token sequences in this vocabulary, and all 8 pass the grammar. Which split a model chooses does not change the automaton state it reaches (test 9), which is why the mask can be a function of the state alone.

**E. Mask size at scale.** For an assumed 128,256-token vocabulary a mask is 4008 words, 15.7 KiB. All 110 states would be 1.68 MiB, small next to the model. The real cost is building a mask: the method here tests every token, about 770,000 character steps in Python for a 6-character average token. No timing was taken, and it is why production systems precompile or use a vocabulary trie.

## 26.8 [FOUNDATIONAL] Failure Modes and Edge Cases

| Situation | What happens | Test |
|---|---|---|
| Output cut off by `max_tokens` | A valid prefix, not valid JSON | 26, experiment C |
| Any state reachable by tokens | Has a legal token and a path to the end | 2, 3 |
| `<eos>` before the document is complete | Cannot happen: the bit is `can_end` | 10 |
| Number at the top level | `<eos>` allowed after the digits | 13 |
| Nesting beyond depth 3 | Rejected | 6 |
| JSON the grammar does not support (negatives, decimals, uppercase, escapes) | Rejected: a documented subset | 8 |
| Same text, different tokenization | Same state, same mask | 9 |
| Vocabulary larger than 32 tokens | Several int32 words, bit 31 stored as a negative number | 15, 18, 19 |
| A row with **no** allowed token | The kernel silently returns token 0, which is disallowed | 22 |
| Greedy limit (temperature near 0) | Highest-logit allowed token | 20 |
| Noise `NaN` or logits `inf` | Not tested | not modelled |

The empty-mask row is the one to act on. The kernel does not check for an all-zero mask, and returning token 0 would feed an illegal token to the automaton, which then returns `None` and crashes at the next lookup. For this grammar test 2 proves it cannot happen. For a new grammar, check the popcount of every mask at compile time, once per state.

**Not modelled.** Schema constraints beyond JSON syntax (types, required keys, enumerations, string patterns); full JSON (arbitrary strings, escapes, Unicode, numbers with signs, fractions and exponents, whitespace); a real tokenizer with byte-level tokens and partial UTF-8 sequences; a real vocabulary (the toy one has 24 tokens); batches where rows use different grammars; speculative decoding combined with masks; the effect on output quality, which needs a real model; GPU-side mask application at a vocabulary of 128,000 tokens, where the kernel's single-block-per-row design would need rework; and all timing.

## 26.9 [FOUNDATIONAL] On Real Hardware (UNTESTED)

Everything above ran on CPU with a toy vocabulary. These commands were written without access to a GPU and have **not been run by the author**; verify them against your installed versions.

```bash
# 1. Correctness on the GPU (no TRITON_INTERPRET)
python3 ch26_grammar_decoding.py --test

# 2. Throughput of the masking kernel at a realistic vocabulary
python3 ch26_grammar_decoding.py --bench
```

**Known gaps before `--bench` means anything:** `masked_sample` allocates its output on the CPU and must create it on `logits.device`; and it launches one program per row with `BLOCK = next_power_of_2(V)`, which is 131,072 for a 128,256-token vocabulary. That may exceed what one program can hold in registers, so first check that it launches at all, then rewrite it to loop over the vocabulary in tiles. Also note that the tests' `decode` loop builds masks on the host, so on a GPU the host-to-device copy of the mask each step is part of the cost. A reasonable hypothesis, not a result: the kernel should be limited by reading the logits and the noise once, and the host-side mask build, not the kernel, should dominate a naive implementation.

## Chapter Summary

Grammar-constrained decoding sets every illegal token's score to minus infinity before sampling. The work is host-side: a pushdown automaton tracks the position in the grammar, each multi-character token is checked by running all its characters through the automaton, and the resulting mask is cached per state and packed into bits. A Triton kernel then applies the bitmask and samples with Gumbel noise. On a toy vocabulary the kernel matched a reference, all 298 finished constrained outputs parsed as JSON while only 4 of 300 unconstrained ones did, and 68 compiled states served about 24,000 lookups. Two outputs were truncated mid-structure, which shows why the guarantee holds only for completed outputs.

## Self-Check Questions

1. Why does JSON need a pushdown automaton rather than a finite automaton?
2. How is a multi-character token such as `":"` checked against the grammar?
3. Why is the mask cached by automaton state, and what does the cache hit count in the output show?
4. Two of 300 constrained outputs were invalid. Why, and how would you prevent it?
5. Why is Gumbel-max sampling equivalent to sampling from the softmax distribution?

## Where We Go Next

Chapter 27 changes the question from "which tokens are allowed" to "how many tokens can we accept per step": speculative decoding, where a small draft model proposes tokens and the large model verifies them in one pass.

## Worked Solutions

1. Closing brackets must match the most recent unclosed opener, which requires remembering an arbitrarily deep sequence of containers. A finite automaton has fixed memory and cannot do this for unbounded nesting.
2. Every character is stepped through the automaton starting from the current state; the token is allowed only if no step fails, so it may legally cross several grammar boundaries in one token.
3. The same states recur constantly, and computing a mask means testing the whole vocabulary. With 68 misses and 23,934 hits, almost every lookup avoided that work.
4. They hit the 80-token limit before the structure closed, so the text was a valid prefix but not a complete document. Reserve budget for closing tokens, force closers near the limit, or count truncation as a failure.
5. Adding independent Gumbel noise to each logit and taking the argmax selects token `i` with probability equal to its softmax probability (the Gumbel-max trick), so it samples the same distribution without computing the normalizing sum.

## Additional Exercises

6. Experiment B shows 110 states but only 19 distinct masks. Propose a smaller cache key that gives identical masks, say what you would project the state onto, and describe how to verify the claim against the 19.
7. The kernel returns token 0 for an all-zero mask. Where would you put the check, and why not in the kernel?
8. The mutation table shows `& inb` and the masked mask-word load are each redundant. Which would you remove, if either?
9. Add negative numbers to the grammar. List the functions that change and the tests that must change with them.
10. Experiment C shows 22 of 100 rows cut off at 10 tokens. Design a way to guarantee completion within the budget and say what it costs.

## Additional Solutions

6. Which tokens are legal now depends on the current mode, any partial literal, the depth (whether another opener fits) and the top of the stack, not on what lies deeper. Key the cache on `(mode, extra, depth, top of stack)`. To verify, group the 110 states by that key and check that every group maps to a single mask; if two states with the same key had different masks, the key is too coarse. The 19 distinct masks give you the target count. Note that the key must be looked up for every state the decode visits, but the compiled mask is shared.
7. On the host, when a mask is first compiled: assert that at least one bit is set. It runs once per state, not once per token, and costs nothing at decode time. A check inside the kernel would run for every row on every step and could at best set a flag, since a kernel cannot raise an exception.
8. Neither, in practice: they protect the same lanes and cost almost nothing. If you remove one, add a comment saying the other is now load-bearing, and keep the V = 70 test, which is the only one that notices when both are gone.
9. `step` must accept `-` at the start of a value and then require a digit; add a mode for "after a minus" so that `-` followed by `0` or a nonzero digit is legal and `-` alone cannot end. `can_end` is unchanged for finished numbers. The vocabulary needs a `-` token. In the tests: the state count (test 1) changes, the start-state hand case gains `-`, and the "documented subset" list must drop `-1` because it is no longer rejected. Add negatives to the generator for the completeness test.
10. Compute, from the state graph, the shortest number of tokens needed to reach a state where `<eos>` is allowed from each state (a breadth-first search backwards from the end states, which the liveness test already builds). When the tokens remaining equal that distance, restrict the mask to tokens that reduce it. This guarantees a complete document as long as the budget at the start is at least the distance from the start state. The cost is a distorted distribution near the end of the budget, since the model is forced to close structures it might have continued.

## Files and Sources

- `examples/references/part8/ch26_grammar_decoding.py`: automaton, mask cache, sampling kernel, decode loop, `main()`, `--test` (29 self-tests, about 15 s), `--extras` (experiments A to E, about 45 s), `--bench` (GPU only, untested, with the gaps noted above).
- Outputs shown were produced with `TRITON_INTERPRET=1` on CPU, Triton 3.9.0 and torch 2.15, on a 24-token toy vocabulary with random logits.
- The 128,256-token vocabulary in experiment E is an assumed Llama-3-sized figure; check your own tokenizer.
