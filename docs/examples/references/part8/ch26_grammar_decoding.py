"""Chapter 26: grammar-constrained decoding with compiled token masks and a bitmask sampling kernel.

Run (CPU, Triton interpreter):
    TRITON_INTERPRET=1 python3 ch26_grammar_decoding.py            (about two minutes)
    TRITON_INTERPRET=1 python3 ch26_grammar_decoding.py --test     (self-tests)
    TRITON_INTERPRET=1 python3 ch26_grammar_decoding.py --extras   (experiments)
Run (GPU only, UNTESTED by the author): python3 ch26_grammar_decoding.py --bench
Toy vocabulary and random logits: this validates the mechanism (masks, validity, cache), not model quality or speed.
"""
import json
import math
import random
import sys
from collections import deque
import torch
import triton
import triton.language as tl

MAXDEPTH = 3
VOCAB = ['{', '}', '[', ']', ',', ':', '"', 'ab', 'cd', 'xyz', '0', '1', '23', '7',
         'true', 'false', 'null', '":"', '","', '{"', '"}', '"]', '[{', '<eos>']
EOS = len(VOCAB) - 1
LIT = {'t': 'true', 'f': 'false', 'n': 'null'}
START = ('VALUE', (), '')


# ------------------------------------------------------------- character-level pushdown automaton
def step(st, c):
    mode, stack, extra = st
    if mode in ('NUM', 'NUM0'):
        if c.isdigit():
            return st if mode == 'NUM' else None          # no digits after a leading 0
        mode = 'AFTER'                                    # number ended; reprocess c as a separator
    if mode in ('VALUE', 'ARRV'):
        if mode == 'ARRV' and c == ']':
            return ('AFTER', stack[:-1], '')
        if c == '{':
            return ('OBJK', stack + ('{',), '') if len(stack) < MAXDEPTH else None
        if c == '[':
            return ('ARRV', stack + ('[',), '') if len(stack) < MAXDEPTH else None
        if c == '"':
            return ('STR', stack, '')
        if c == '0':
            return ('NUM0', stack, '')
        if c in '123456789':
            return ('NUM', stack, '')
        if c in LIT:
            return ('LIT', stack, LIT[c][1:])
        return None
    if mode == 'OBJK':
        return ('AFTER', stack[:-1], '') if c == '}' else (('KEYSTR', stack, '') if c == '"' else None)
    if mode == 'OBJK2':
        return ('KEYSTR', stack, '') if c == '"' else None
    if mode == 'KEYSTR':
        return st if c.isalpha() and c.islower() else (('COLON', stack, '') if c == '"' else None)
    if mode == 'COLON':
        return ('VALUE', stack, '') if c == ':' else None
    if mode == 'STR':
        return st if c.isalpha() and c.islower() else (('AFTER', stack, '') if c == '"' else None)
    if mode == 'LIT':
        if c != extra[0]:
            return None
        return ('AFTER', stack, '') if len(extra) == 1 else ('LIT', stack, extra[1:])
    if mode == 'AFTER':
        if not stack:
            return None
        if stack[-1] == '{':
            return ('OBJK2', stack, '') if c == ',' else (('AFTER', stack[:-1], '') if c == '}' else None)
        return ('VALUE', stack, '') if c == ',' else (('AFTER', stack[:-1], '') if c == ']' else None)
    return None


def run_token(st, tok):
    for c in tok:
        st = step(st, c)
        if st is None:
            return None
    return st


def can_end(st):
    return st[1] == () and st[0] in ('AFTER', 'NUM', 'NUM0')


# ------------------------------------------------------------- mask compilation + cache
class MaskCache:
    def __init__(self):
        self.cache, self.hits, self.misses = {}, 0, 0

    def get(self, st):
        if st in self.cache:
            self.hits += 1
            return self.cache[st]
        self.misses += 1
        ok = [False] * len(VOCAB)
        for i, tok in enumerate(VOCAB[:-1]):
            ok[i] = run_token(st, tok) is not None
        ok[EOS] = can_end(st)
        words = (len(VOCAB) + 31) // 32
        packed = [0] * words
        for i, a in enumerate(ok):
            if a:
                packed[i // 32] |= 1 << (i % 32)
        packed = [w - (1 << 32) if w >= (1 << 31) else w for w in packed]    # to signed int32
        self.cache[st] = torch.tensor(packed, dtype=torch.int32)
        return self.cache[st]


# ------------------------------------------------------------- GPU kernel: apply bitmask, Gumbel-max sample
@triton.jit
def masked_sample_kernel(logits_ptr, mask_ptr, noise_ptr, out_ptr, V, n_words, inv_temp,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    inb = i < V
    word = tl.load(mask_ptr + row * n_words + i // 32, mask=inb, other=0)
    allowed = ((word >> (i % 32)) & 1) == 1
    x = tl.load(logits_ptr + row * V + i, mask=inb, other=0.0).to(tl.float32) * inv_temp
    g = tl.load(noise_ptr + row * V + i, mask=inb, other=0.0)          # Gumbel noise, supplied by host
    score = tl.where(allowed & inb, x + g, -1.0e30)                    # disallowed tokens can never win
    tl.store(out_ptr + row, tl.argmax(score, axis=0))


def masked_sample(logits, masks, noise, temp=1.0):
    B, V = logits.shape
    out = torch.empty(B, dtype=torch.int64)
    masked_sample_kernel[(B,)](logits, masks, noise, out, V, masks.shape[1], 1.0 / temp,
                               BLOCK=triton.next_power_of_2(V))
    return out


def reference_sample(logits, masks, noise, temp=1.0):
    B, V = logits.shape
    i = torch.arange(V)
    allowed = ((masks[:, i // 32] >> (i % 32)) & 1) == 1
    score = torch.where(allowed, logits / temp + noise, torch.tensor(-1e30))
    return score.argmax(-1)


# ------------------------------------------------------------- decode loop
def decode(batch, max_tokens, constrained, seed, cache):
    g = torch.Generator().manual_seed(seed)
    V = len(VOCAB)
    states = [START] * batch
    outs = [[] for _ in range(batch)]
    done = [False] * batch
    for _ in range(max_tokens):
        logits = torch.randn(batch, V, generator=g)
        u = torch.rand(batch, V, generator=g).clamp(1e-9, 1 - 1e-9)
        noise = -torch.log(-torch.log(u))
        if constrained:
            masks = torch.stack([cache.get(s) for s in states])
        else:
            masks = torch.full((batch, 1), -1, dtype=torch.int32)       # all bits set: nothing masked
            masks = masks.expand(batch, (V + 31) // 32).contiguous()
        toks = masked_sample(logits, masks, noise)
        for b in range(batch):
            if done[b]:
                continue
            t = int(toks[b])
            if t == EOS:
                done[b] = True
                continue
            outs[b].append(VOCAB[t])
            if constrained:
                states[b] = run_token(states[b], VOCAB[t])
        if all(done):
            break
    return ["".join(o) for o in outs], done


def is_valid(s):
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def main():
    cache = MaskCache()
    # 1) kernel equals the torch reference, including a state with many masked tokens
    st = run_token(START, '{"ab":')
    m = torch.stack([cache.get(START), cache.get(st)])
    lg = torch.randn(2, len(VOCAB))
    nz = -torch.log(-torch.log(torch.rand(2, len(VOCAB)).clamp(1e-9, 1 - 1e-9)))
    print("kernel == reference?", bool((masked_sample(lg, m, nz) == reference_sample(lg, m, nz)).all()))

    # 2) validity
    N, T = 300, 80
    out_c, done_c = decode(N, T, True, 1, cache)
    complete = [o for o, d in zip(out_c, done_c) if d]
    print(f"constrained: finished {len(complete)}/{N}; valid among finished = "
          f"{sum(map(is_valid, complete))}/{len(complete)}; truncated at max_tokens = {N - len(complete)}")
    print(f"constrained, ALL outputs (incl. truncated) valid: {sum(map(is_valid, out_c))}/{N}")
    out_u, _ = decode(N, T, False, 1, cache)
    print(f"unconstrained: valid JSON = {sum(map(is_valid, out_u))}/{N}")
    print(f"mask cache: {len(cache.cache)} distinct states compiled, hits={cache.hits}, misses={cache.misses}")
    print("sample constrained outputs:", complete[:3])


# ------------------------------------------------------------- self-tests
def _reach(vocab=None):
    """Explore every state reachable from START by tokens of the vocabulary: (states, edges)."""
    vocab = VOCAB if vocab is None else vocab
    seen, dq, edges = {START}, deque([START]), {}
    while dq:
        st = dq.popleft()
        edges[st] = []
        for tok in vocab[:-1]:
            nx = run_token(st, tok)
            if nx is not None:
                edges[st].append(nx)
                if nx not in seen:
                    seen.add(nx); dq.append(nx)
    return seen, edges


def _accepts(text):
    st = START
    for c in text:
        st = step(st, c)
        if st is None:
            return False
    return can_end(st)


def _unpack(mask_row, V):
    words = [int(w) & 0xFFFFFFFF for w in mask_row.tolist()]
    return [bool((words[i // 32] >> (i % 32)) & 1) for i in range(V)]


def _gen_json(rng, depth=0):
    r = rng.random()
    if depth < 3 and r < 0.35:
        return [_gen_json(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    if depth < 3 and r < 0.6:
        keys = rng.sample(["a", "b", "xy", "key", "z", "ab"], rng.randint(0, 3))
        return {k: _gen_json(rng, depth + 1) for k in keys}
    return rng.choice([0, 1, 7, 23, 100, 305, True, False, None, "", "ab", "xyz", "hello"])


def _pack_words(ok):
    words = [0] * ((len(ok) + 31) // 32)
    for i, a in enumerate(ok):
        if a:
            words[i // 32] |= 1 << (i % 32)
    return torch.tensor([w - (1 << 32) if w >= (1 << 31) else w for w in words], dtype=torch.int32)


def run_tests():
    global VOCAB, EOS
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # ---------------- the automaton (pure Python)
    seen, edges = _reach()
    check("exploring every token from every state reaches 110 states (a regression value: it changes if the grammar or vocabulary does)", len(seen) == 110)
    check("no dead state: every reachable state allows at least one token (counting <eos>)", all(edges[s] or can_end(s) for s in seen))
    rev = {s: [] for s in seen}
    for s_, ns in edges.items():
        for n in ns:
            rev[n].append(s_)
    live, dq = {s_ for s_ in seen if can_end(s_)}, deque(s_ for s_ in seen if can_end(s_))
    while dq:
        x = dq.popleft()
        for q_ in rev[x]:
            if q_ not in live:
                live.add(q_); dq.append(q_)
    check("no trap state: from every reachable state some token sequence reaches a state where <eos> is allowed", live == seen)
    rng = random.Random(0)
    vals = [_gen_json(rng) for _ in range(400)]
    check("completeness: 400 random JSON values in the supported subset (depth <= 3), written compactly, are all accepted character by character",
          all(_accepts(json.dumps(v, separators=(",", ":"))) for v in vals))
    ok_all, finished = True, 0
    for _ in range(3000):
        st, text = START, ""
        for _ in range(60):
            allowed = [t for t in VOCAB[:-1] if run_token(st, t) is not None]
            if can_end(st) and (not allowed or rng.random() < 0.3):
                finished += 1
                ok_all &= is_valid(text)
                break
            t = rng.choice(allowed)
            text += t; st = run_token(st, t)
    check("soundness: 3000 random walks through allowed tokens that end at <eos> are all valid JSON (finished: %d)" % finished, ok_all and finished > 1500)
    should_reject = ["01", "[1,]", "{,}", '{"a":1,}', "[", '{"a"}', "nul", "truee", "[1 2]", '{"a":}', "]", "}", ",", "1,", '"ab', "[[[[1]]]]", "[[[["]
    check("17 malformed strings (leading zero, trailing comma, unclosed, depth 4, ...) are rejected", not any(_accepts(t) for t in should_reject))
    should_accept = ["0", "10", "[]", "{}", "[[]]", '{"a":[1,2,{"b":null}]}', "true", '"abc"', '""', "[[[1]]]", "[[[]]]", '{"":1}']
    check("12 well-formed strings (including depth 3, empty containers and the empty key) are accepted", all(_accepts(t) for t in should_accept))
    subset = ["-1", "1.5", "1e3", '"A"', '"a b"', "[ 1]", '"a\\n"', '{"A":1}', '{"a1":1}']
    check("documented subset: 9 strings that real JSON accepts are rejected here (negative, decimal, exponent, uppercase, space, escape, digit in key)",
          all(is_valid(t) and not _accepts(t) for t in subset))
    for a_, b_ in ((['{', '"', 'ab', '"', ':', '1', '}'], ['{"', 'ab', '"', ':', '1', '}']), (['[{', '}', ']'], ['[', '{', '}', ']'])):
        sa = sb = START
        for t in a_:
            sa = run_token(sa, t)
        for t in b_:
            sb = run_token(sb, t)
        same = sa == sb and sa is not None and can_end(sa)
        if not same:
            break
    check("tokenization independence: two different token splits of the same text reach the identical final state", same)

    # ---------------- mask compilation and cache
    cache = MaskCache()
    exact = True
    for st in seen:
        bits = _unpack(cache.get(st), len(VOCAB))
        for i, tok in enumerate(VOCAB[:-1]):
            cur, alive = st, True
            for c in tok:
                cur = step(cur, c)
                if cur is None:
                    alive = False; break
            exact &= (bits[i] == alive)
        exact &= (bits[EOS] == can_end(st))
    check("for all 110 states, mask bit i equals 'every character of token i is accepted' (recomputed independently with step), and the <eos> bit equals can_end", exact)
    start_set = {VOCAB[i] for i, b in enumerate(_unpack(cache.get(START), len(VOCAB))) if b}
    check("hand case: at the start the allowed tokens are exactly { [ \" 0 1 23 7 true false null {\" [{ (12 tokens)",
          start_set == {'{', '[', '"', '0', '1', '23', '7', 'true', 'false', 'null', '{"', '[{'})
    st_num = run_token(START, '{"ab":1')
    set_num = {VOCAB[i] for i, b in enumerate(_unpack(cache.get(st_num), len(VOCAB))) if b}
    check("hand case: after {\"ab\":1 only more digits, a comma or a closing brace are allowed, not <eos> and not a quote",
          set_num == {'0', '1', '23', '7', ',', '}'})
    set_top = {VOCAB[i] for i, b in enumerate(_unpack(cache.get(run_token(START, '1')), len(VOCAB))) if b}
    check("hand case: after a top-level 1, <eos> is allowed because a number may end there, and ',' is not", set_top == {'0', '1', '23', '7', '<eos>'})
    c2 = MaskCache()
    a1 = c2.get(START); h0, m0 = c2.hits, c2.misses
    a2 = c2.get(START)
    check("cache: the first lookup is a miss, the second returns the same tensor and counts a hit", (h0, m0) == (0, 1) and a2 is a1 and (c2.hits, c2.misses) == (1, 1))
    # larger vocabulary: three 32-bit words and the signed-int32 conversion for bit 31
    old_v, old_e = VOCAB, EOS
    try:
        VOCAB = old_v[:-1] + ["q" * k for k in range(1, 48)] + ["<eos>"]
        EOS = len(VOCAB) - 1
        c3 = MaskCache()
        st_str = run_token(START, '"')
        row = c3.get(st_str)
        bits = _unpack(row, len(VOCAB))
        letters = [i for i, t in enumerate(VOCAB[:-1]) if t.isalpha() and t.islower()]
        big_ok = row.shape == (3,) and row.dtype == torch.int32 and all(bits[i] for i in letters) and bits[31] and bool((row < 0).any())
    finally:
        VOCAB, EOS = old_v, old_e
    check("a 71-token vocabulary packs into 3 int32 words, with a negative word when bit 31 is set, and unpacks exactly", big_ok)

    # ---------------- the sampling kernel
    g = torch.Generator().manual_seed(11)
    V = len(VOCAB)

    def noise_of(B, V_):
        u = torch.rand(B, V_, generator=g).clamp(1e-9, 1 - 1e-9)
        return -torch.log(-torch.log(u))

    same_all, in_mask = True, True
    for temp in (1.0, 0.5, 2.0):
        B = 48
        rows = []
        for _ in range(B):
            ok = [rng.random() < 0.4 for _ in range(V)]
            ok[rng.randrange(V)] = True
            rows.append(_pack_words(ok))
        masks = torch.stack(rows)
        lg = torch.randn(B, V, generator=g)
        nz = noise_of(B, V)
        got, want = masked_sample(lg, masks, nz, temp), reference_sample(lg, masks, nz, temp)
        same_all &= bool((got == want).all())
        in_mask &= all(_unpack(masks[b], V)[int(got[b])] for b in range(B))
    check("kernel == torch reference on 3 batches of 48 random masks at temperature 1, 0.5 and 2", same_all)
    check("the kernel never returns a token whose mask bit is clear (when at least one bit is set)", in_mask)
    Vb = 70
    single = torch.stack([_pack_words([i == r for i in range(Vb)]) for r in range(Vb)])
    got = masked_sample(torch.randn(Vb, Vb, generator=g), single, noise_of(Vb, Vb))
    check("V = 70 (3 words, BLOCK 128): a mask with only token r allowed returns r for every r, across the bit-31/32/63/64 word boundaries", got.tolist() == list(range(Vb)))
    ok_v = True
    for Vx in (32, 33):
        mk = torch.stack([_pack_words([rng.random() < 0.5 or i == 0 for i in range(Vx)]) for _ in range(20)])
        lgx, nzx = torch.randn(20, Vx, generator=g), noise_of(20, Vx)
        ok_v &= bool((masked_sample(lgx, mk, nzx) == reference_sample(lgx, mk, nzx)).all())
    check("V = 32 (exactly one word) and V = 33 (spills into a second word) match the reference", ok_v)
    mk = torch.stack([_pack_words([rng.random() < 0.5 or i == 3 for i in range(V)]) for _ in range(30)])
    lg = torch.randn(30, V, generator=g)
    cold = masked_sample(lg, mk, noise_of(30, V), temp=1e-3)
    greedy = torch.where(torch.stack([torch.tensor(_unpack(mk[b], V)) for b in range(30)]), lg, torch.tensor(-1e30)).argmax(-1)
    check("temperature 1e-3 is greedy: the kernel returns the highest-logit ALLOWED token", torch.equal(cold, greedy))
    nz0 = torch.zeros(30, V)
    check("with zero noise the kernel is a masked argmax of the logits", torch.equal(masked_sample(lg, mk, nz0), reference_sample(lg, mk, nz0)))
    emp = torch.zeros(1, 1, dtype=torch.int32)
    out_e = masked_sample(torch.randn(1, V, generator=g), emp, noise_of(1, V))
    check("documented limitation: a row with NO allowed token silently returns token 0, which is disallowed (the caller must prevent empty masks)", int(out_e[0]) == 0)
    lg1 = torch.randn(1, V, generator=g).expand(200000, V)
    ok1 = [rng.random() < 0.5 for _ in range(V)]; ok1[0] = True
    m1 = _pack_words(ok1)[None, :].expand(200000, 1)
    draws = reference_sample(lg1, m1, noise_of(200000, V))
    freq = torch.bincount(draws, minlength=V).float() / 200000
    p = torch.where(torch.tensor(ok1), lg1[0].exp(), torch.zeros(V)); p = p / p.sum()
    check("Gumbel-max sampling follows softmax over the allowed tokens: 200,000 draws within 0.01 of the exact probabilities", float((freq - p).abs().max()) < 0.01)
    snap = (lg.clone(), mk.clone())
    a_, b_ = masked_sample(lg, mk, nz0), masked_sample(lg, mk, nz0)
    check("the kernel is deterministic and does not modify its inputs", torch.equal(a_, b_) and torch.equal(snap[0], lg) and torch.equal(snap[1], mk))

    # ---------------- the decode loop
    try:
        cch = MaskCache()
        outs, done = decode(40, 50, True, 3, cch)
        fin = [o for o, d in zip(outs, done) if d]
        check("constrained decode, 40 rows x 50 tokens: every FINISHED output is valid JSON (finished: %d/40)" % len(fin), len(fin) > 20 and all(map(is_valid, fin)))
        live_ok = True
        for o in outs:
            st_ = START
            for ch in o:
                st_ = step(st_, ch)
                if st_ is None:
                    break
            live_ok &= st_ is not None and st_ in live
        check("every output, including the ones cut off at max_tokens, is a live prefix: the automaton is still alive and could still finish", live_ok)
        check("the mask cache holds only states that are reachable (at most 110)", len(cch.cache) <= 110 and set(cch.cache) <= seen)
        outs_u, _ = decode(40, 50, False, 3, MaskCache())
        check("unconstrained decode of the same shape: fewer than half the outputs are valid JSON", sum(map(is_valid, outs_u)) < 20)
        o_a, _ = decode(12, 30, True, 5, MaskCache()); o_b, _ = decode(12, 30, True, 5, MaskCache()); o_c, _ = decode(12, 30, True, 6, MaskCache())
        check("the same seed reproduces the outputs exactly, and a different seed changes them", o_a == o_b and o_a != o_c)
    except Exception as exc:
        check("the decode loop ran to completion (it raised %s: %s)" % (type(exc).__name__, str(exc)[:60]), False)

    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g_ for _, g_ in results)}/{len(results)} self-tests pass")
    return all(g_ for _, g_ in results)



# ------------------------------------------------------------- extras
def extras():
    global MAXDEPTH
    seen, edges = _reach()
    cache = MaskCache()
    masks = {tuple(cache.get(st).tolist()) for st in seen}
    counts = sorted(sum(1 for t in VOCAB[:-1] if run_token(st, t) is not None) + int(can_end(st)) for st in seen)
    print("Experiment A: size of the state space (toy vocabulary, 24 tokens, MAXDEPTH 3)")
    print(f"reachable automaton states: {len(seen)}; distinct masks: {len(masks)}; bytes per mask: {cache.get(START).numel() * 4}")
    print(f"allowed tokens per state: min {counts[0]}, median {counts[len(counts)//2]}, mean {sum(counts)/len(counts):.2f}, max {counts[-1]} of {len(VOCAB)}; "
          f"states with exactly one allowed token: {sum(c == 1 for c in counts)}")
    print("\nExperiment B: how the state count grows with the nesting limit (same vocabulary)")
    print("MAXDEPTH  states")
    old = MAXDEPTH
    try:
        for d in range(1, 7):
            MAXDEPTH = d
            print(f"{d:8d}  {len(_reach()[0]):6d}")
    finally:
        MAXDEPTH = old
    print("the stack is part of the state, so the count roughly doubles per level: a cache keyed by full state does not stay small for deep grammars")
    print("\nExperiment C: validity against the output budget (100 rows per cell)")
    print("max_tokens  constrained: finished  valid-of-finished  cut off   unconstrained: valid")
    for T in (10, 30, 60):
        oc, dc = decode(100, T, True, 7, MaskCache())
        fin = [o for o, d in zip(oc, dc) if d]
        ou, _ = decode(100, T, False, 7, MaskCache())
        print(f"{T:10d}  {len(fin):19d}  {sum(map(is_valid, fin)):10d}/{len(fin):<5d} {100 - len(fin):7d}   {sum(map(is_valid, ou)):17d}")
    print("a cut-off output is a valid PREFIX, not valid JSON: the guarantee needs enough budget to reach the end")
    print("\nExperiment D: tokenization ambiguity (number of token sequences that spell the same text)")
    def ways(text, grammar):
        memo = {}
        def go(i, st):
            if i == len(text):
                return 1
            key = (i, st if grammar else None)
            if key in memo:
                return memo[key]
            n = 0
            for t in VOCAB[:-1]:
                if text.startswith(t, i):
                    nx = run_token(st, t) if grammar else st
                    if nx is not None:
                        n += go(i + len(t), nx)
            memo[key] = n
            return n
        return go(0, START)
    for text in ('{"ab":1}', '{"ab":"cd"}', '[{"xyz":[23]}]'):
        print(f"{text:18s} vocabulary only: {ways(text, False):3d}   accepted by the grammar: {ways(text, True):3d}")
    print("the mask must therefore be a function of the automaton state alone, never of how the text was split into tokens")
    print("\nExperiment E: mask size for a large vocabulary (assumed 128,256 tokens, like a Llama-3 tokenizer)")
    V = 128256
    words = (V + 31) // 32
    print(f"{words} int32 words = {words*4} bytes ({words*4/1024:.1f} KiB) per mask; {len(masks)} distinct masks would need {len(masks)*words*4/1024:.0f} KiB, "
          f"all {len(seen)} states {len(seen)*words*4/2**20:.2f} MiB")
    print(f"compiling one mask by this method runs the automaton over every token: {V} tokens x (average token length, say 6) = {V*6:,} character steps in Python")


# ------------------------------------------------------------- GPU benchmark (UNTESTED: written without access to a GPU)
def bench():
    if not torch.cuda.is_available():
        print("--bench needs a CUDA GPU; none found. Nothing was measured.")
        return
    from triton.testing import do_bench
    V = 128256
    words = (V + 31) // 32
    print("NOTE: masked_sample allocates its output on the CPU and runs one program per row with BLOCK = next_power_of_2(V) = 131072: fix the device and check it launches at all.")
    print("rows  time_us  GB/s (logits + noise read once)")
    for B in (1, 8, 64, 256):
        lg = torch.randn(B, V, device="cuda"); nz = torch.randn(B, V, device="cuda")
        mk = torch.randint(-2**31, 2**31 - 1, (B, words), dtype=torch.int32, device="cuda")
        t = do_bench(lambda: masked_sample(lg, mk, nz), warmup=25, rep=100)
        print(f"{B:4d}  {t*1e3:7.1f}  {2*B*V*4/(t*1e-3)/1e9:7.0f}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--bench" in sys.argv:
        bench(); sys.exit(0)
    main()
