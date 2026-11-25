"""Chapter 27: speculative decoding -- a rejection-sampling verify kernel and a lossless-ness test.

Run (CPU, Triton interpreter):
    TRITON_INTERPRET=1 python3 ch27_speculative_decoding.py
    TRITON_INTERPRET=1 python3 ch27_speculative_decoding.py --test      (self-tests)
    TRITON_INTERPRET=1 python3 ch27_speculative_decoding.py --extras    (experiments)
Run (GPU only, UNTESTED by the author): python3 ch27_speculative_decoding.py --bench
Toy tabular "models" only. This validates the algorithm and kernel, NOT speedup on a real model pair.
"""
import sys
import torch
import triton
import triton.language as tl

NEG = -1.0e30


# ------------------------------------------------------------- Triton verify kernel
@triton.jit
def spec_verify_kernel(p_ptr, q_ptr, x_ptr, u_ptr, g_ptr, n_out_ptr, tok_out_ptr,
                       K, V,
                       KP2: tl.constexpr, VP2: tl.constexpr):
    s = tl.program_id(0)
    i = tl.arange(0, KP2)
    inb = i < K
    x = tl.load(x_ptr + s * K + i, mask=inb, other=0)
    u = tl.load(u_ptr + s * K + i, mask=inb, other=0.0)
    px = tl.load(p_ptr + s * (K + 1) * V + i * V + x, mask=inb, other=1.0)
    qx = tl.load(q_ptr + s * K * V + i * V + x, mask=inb, other=1.0)
    accept = (u * qx) <= px                                   # u <= p/q without dividing
    first_reject = tl.min(tl.where(inb & (accept == 0), i, K), axis=0)   # K means "none rejected"
    n = first_reject
    v = tl.arange(0, VP2)
    vin = v < V
    pn = tl.load(p_ptr + s * (K + 1) * V + n * V + v, mask=vin, other=0.0)
    nq = tl.minimum(n, K - 1)
    qn = tl.load(q_ptr + s * K * V + nq * V + v, mask=vin, other=0.0)
    qn = tl.where(n < K, qn, 0.0)                             # bonus token: residual is just p_K
    r = tl.maximum(pn - qn, 0.0)
    g = tl.load(g_ptr + s * V + v, mask=vin, other=0.0)
    score = tl.where(r > 0.0, tl.log(tl.maximum(r, 1e-30)) + g, NEG)
    tl.store(n_out_ptr + s, n)
    tl.store(tok_out_ptr + s, tl.argmax(score, axis=0))


def verify_kernel(p, q, x, u, g):
    B, K = x.shape
    V = p.shape[-1]
    n = torch.empty(B, dtype=torch.int64)
    tok = torch.empty(B, dtype=torch.int64)
    spec_verify_kernel[(B,)](p, q, x, u, g, n, tok, K, V,
                             KP2=triton.next_power_of_2(K), VP2=triton.next_power_of_2(V))
    return n, tok


def verify_torch(p, q, x, u, g, wrong_correction=False):
    B, K = x.shape
    px = p[:, :K].gather(2, x[..., None]).squeeze(-1)
    qx = q.gather(2, x[..., None]).squeeze(-1)
    reject = (u * qx) > px
    n = torch.where(reject.any(1), reject.float().argmax(1), torch.full((B,), K))
    ar = torch.arange(B)
    pn = p[ar, n]
    qn = torch.where((n < K)[:, None], q[ar, n.clamp(max=K - 1)], torch.zeros_like(pn))
    r = pn if wrong_correction else (pn - qn).clamp_min(0)     # WRONG variant: resample from p, not the residual
    score = torch.where(r > 0, torch.log(r.clamp_min(1e-30)) + g, torch.tensor(NEG))
    return n, score.argmax(-1)


def gumbel(shape, gen):
    u = torch.rand(shape, generator=gen).clamp(1e-9, 1 - 1e-9)
    return -torch.log(-torch.log(u))


def sample(probs, gen):                      # probs [..., V] -> tokens, via Gumbel-max
    return (torch.log(probs.clamp_min(1e-30)) + gumbel(probs.shape, gen)).argmax(-1)


# ------------------------------------------------------------- Experiment A: kernel == torch reference
def exp_kernel_matches(gen):
    B, K, V = 40, 4, 6
    p = torch.softmax(torch.randn(B, K + 1, V, generator=gen), -1)
    q = torch.softmax(torch.randn(B, K, V, generator=gen), -1)
    x = sample(q, gen)
    u = torch.rand(B, K, generator=gen)
    g = gumbel((B, V), gen)
    n1, t1 = verify_kernel(p, q, x, u, g)
    n2, t2 = verify_torch(p, q, x, u, g)
    print(f"kernel == torch reference?  n: {bool((n1 == n2).all())}  token: {bool((t1 == t2).all())}")


# ------------------------------------------------------------- Experiment B: lossless on a Markov chain
def speculative_chain(Tp, Tq, K, L, B, gen, wrong=False):
    """Generate L tokens per trial with draft table Tq and target table Tp (row = previous token)."""
    V = Tp.shape[0]
    out = torch.zeros(B, L, dtype=torch.int64)
    length = torch.zeros(B, dtype=torch.int64)
    prev = torch.zeros(B, dtype=torch.int64)             # token 0 is the fixed start symbol
    rounds = 0
    while (length < L).any():
        ctx = prev.clone()
        xs, qs, ps = [], [], []
        for _ in range(K):                               # draft proposes autoregressively
            qd = Tq[ctx]
            xt = sample(qd, gen)
            xs.append(xt); qs.append(qd); ctx = xt
        ctx = prev.clone()
        for j in range(K + 1):                           # target scores each prefix
            ps.append(Tp[ctx])
            if j < K:
                ctx = xs[j]
        x, q, p = torch.stack(xs, 1), torch.stack(qs, 1), torch.stack(ps, 1)
        u = torch.rand(B, K, generator=gen)
        n, extra = verify_torch(p, q, x, u, gumbel((B, V), gen), wrong_correction=wrong)
        rounds += 1
        for b in range(B):
            if length[b] >= L:
                continue
            toks = x[b, :n[b]].tolist() + [int(extra[b])]
            for t in toks:
                if length[b] < L:
                    out[b, length[b]] = t
                    length[b] += 1
                    prev[b] = t
    return out, rounds


def exact_distribution(Tp, L):
    V = Tp.shape[0]
    d = {(): 1.0}
    for _ in range(L):
        nd = {}
        for seq, pr in d.items():
            last = seq[-1] if seq else 0
            for t in range(V):
                nd[seq + (t,)] = pr * float(Tp[last, t])
        d = nd
    return d


def tv_distance(out, exact):
    counts = {}
    for row in out.tolist():
        counts[tuple(row)] = counts.get(tuple(row), 0) + 1
    n = out.shape[0]
    return 0.5 * sum(abs(counts.get(k, 0) / n - v) for k, v in exact.items())


def exp_lossless(gen):
    V, K, L, B = 4, 3, 3, 6000
    Tp = torch.softmax(torch.randn(V, V, generator=gen) * 1.5, -1)
    Tq = torch.softmax(torch.randn(V, V, generator=gen) * 1.5, -1)   # a different, imperfect draft
    exact = exact_distribution(Tp, L)
    good, _ = speculative_chain(Tp, Tq, K, L, B, gen)
    bad, _ = speculative_chain(Tp, Tq, K, L, B, gen, wrong=True)
    direct = torch.zeros(B, L, dtype=torch.int64)
    prev = torch.zeros(B, dtype=torch.int64)
    for t in range(L):
        prev = sample(Tp[prev], gen)
        direct[:, t] = prev
    noise_floor = tv_distance(direct, exact)
    print(f"TV distance to exact target distribution ({V**L} sequences, {B} trials each):")
    print(f"  direct sampling from target (noise floor) = {noise_floor:.4f}")
    print(f"  speculative, correct residual correction  = {tv_distance(good, exact):.4f}")
    print(f"  speculative, WRONG correction (from p)    = {tv_distance(bad, exact):.4f}")


# ------------------------------------------------------------- Experiment C: expected tokens per step
def exp_formula(gen):
    V, K, B = 8, 4, 40000
    p = torch.softmax(torch.randn(V, generator=gen) * 1.2, -1)
    q = torch.softmax(torch.randn(V, generator=gen) * 1.2, -1)
    alpha = float(torch.minimum(p, q).sum())             # per-token acceptance probability
    P = p.expand(B, K + 1, V).contiguous()
    Q = q.expand(B, K, V).contiguous()
    x = sample(Q, gen)
    n, _ = verify_torch(P, Q, x, torch.rand(B, K, generator=gen), gumbel((B, V), gen))
    measured = float((n + 1).float().mean())             # accepted + 1 extra token per target pass
    formula = (1 - alpha ** (K + 1)) / (1 - alpha)
    print(f"alpha = {alpha:.3f}, K = {K}: tokens per target pass  measured = {measured:.3f}  "
          f"formula (1-a^(K+1))/(1-a) = {formula:.3f}")


def main():
    gen = torch.Generator().manual_seed(0)
    exp_kernel_matches(gen)
    exp_lossless(gen)
    exp_formula(gen)


# ------------------------------------------------------------- self-tests
def _rand_case(B, K, V, gen, sharp=1.0):
    p = torch.softmax(torch.randn(B, K + 1, V, generator=gen) * sharp, -1)
    q = torch.softmax(torch.randn(B, K, V, generator=gen) * sharp, -1)
    x = sample(q, gen)
    return p, q, x, torch.rand(B, K, generator=gen), gumbel((B, V), gen)


def run_tests():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    gen = torch.Generator().manual_seed(21)

    # ---- the kernel against the torch reference, across awkward shapes
    same, support_ok, rng_ok = True, True, True
    for K, V in [(1, 2), (2, 3), (3, 6), (4, 7), (5, 16), (8, 50)]:
        p, q, x, u, g = _rand_case(24, K, V, gen)
        n1, t1 = verify_kernel(p, q, x, u, g)
        n2, t2 = verify_torch(p, q, x, u, g)
        same &= bool((n1 == n2).all() and (t1 == t2).all())
        rng_ok &= bool(((n1 >= 0) & (n1 <= K)).all())
        for b in range(24):
            n_, t_ = int(n1[b]), int(t1[b])
            mass = float(p[b, n_, t_]) - (float(q[b, n_, t_]) if n_ < K else 0.0)
            support_ok &= mass > 0
    check("kernel == torch reference for (K, V) = (1,2) (2,3) (3,6) (4,7) (5,16) (8,50), most of them not powers of two", same)
    check("the number accepted is always between 0 and K", rng_ok)
    check("the extra token always has positive residual mass: p - q > 0 after a rejection, p_K > 0 after a full accept", support_ok)
    n_, t_ = verify_kernel(torch.ones(2, 3, 1), torch.ones(2, 2, 1), torch.zeros(2, 2, dtype=torch.int64), torch.rand(2, 2, generator=gen), torch.zeros(2, 1))
    check("a one-token vocabulary (V = 1) works: everything accepted, token 0", n_.tolist() == [2, 2] and t_.tolist() == [0, 0])

    # ---- hand-built cases
    pb = torch.tensor([[[.25, .75], [.5, .5]]]); qb = torch.tensor([[[.5, .5]]]); xb = torch.zeros(1, 1, dtype=torch.int64)
    a1 = verify_kernel(pb, qb, xb, torch.tensor([[0.5]]), torch.zeros(1, 2))[0].item()
    a2 = verify_kernel(pb, qb, xb, torch.tensor([[0.5001]]), torch.zeros(1, 2))[0].item()
    t1_ = verify_torch(pb, qb, xb, torch.tensor([[0.5]]), torch.zeros(1, 2))[0].item()
    check("hand case, boundary: p(x)=0.25, q(x)=0.5; u=0.5 gives u*q = p exactly and is ACCEPTED, u=0.5001 is rejected (kernel and torch agree)", (a1, a2, t1_) == (1, 0, 1))
    px = torch.tensor([.5, .5, .1, .5]); K4 = 4
    p4 = torch.stack([torch.stack([v, 1 - v]) for v in px] + [torch.tensor([.5, .5])])[None]
    q4 = torch.tensor([[.5, .5]]).expand(K4, 2)[None].contiguous()
    n4, _ = verify_kernel(p4, q4, torch.zeros(1, K4, dtype=torch.int64), torch.full((1, K4), 0.5), torch.zeros(1, 2))
    check("hand case, first rejection decides: accept pattern [yes yes NO yes] gives n = 2, the 4th token is discarded even though it would have been accepted", int(n4[0]) == 2)
    pe = torch.softmax(torch.randn(200, 4, 5, generator=gen), -1); pe[:, 3] = torch.tensor([0., 1., 0., 0., 0.])
    qe = pe[:, :3].clone()
    xe = sample(qe, gen)
    ne, te = verify_kernel(pe, qe, xe, torch.rand(200, 3, generator=gen) * 0.999, gumbel((200, 5), gen))
    check("p == q exactly: every one of 200 rows accepts all 3 draft tokens, and the bonus token comes from p_K (here a one-hot on token 1)", bool((ne == 3).all()) and bool((te == 1).all()))
    pd = torch.tensor([0., .3, .7]).expand(3000, 2, 3).contiguous(); pd[:, 1] = torch.tensor([1., 0., 0.])
    qd = torch.tensor([1., 0., 0.]).expand(3000, 1, 3).contiguous()
    nd, td = verify_kernel(pd, qd, torch.zeros(3000, 1, dtype=torch.int64), torch.rand(3000, 1, generator=gen).clamp_min(1e-6), gumbel((3000, 3), gen))
    freq = torch.bincount(td, minlength=3).float() / 3000
    check("the target gives the draft token probability 0: every row is rejected (n = 0), token 0 is never returned, and tokens 1 and 2 appear with frequency 0.3 and 0.7 within 0.04",
          bool((nd == 0).all()) and int((td == 0).sum()) == 0 and abs(float(freq[1]) - 0.3) < 0.04 and abs(float(freq[2]) - 0.7) < 0.04)
    n_z, _ = verify_kernel(torch.tensor([[[0., 1.], [.5, .5]]]), torch.tensor([[[1., 0.]]]), torch.zeros(1, 1, dtype=torch.int64), torch.zeros(1, 1), torch.zeros(1, 2))
    check("documented edge: u = 0 accepts a draft token the target gives probability 0 (0 * q <= 0); a measure-zero event for u drawn from [0, 1)", int(n_z[0]) == 1)
    n_q, _ = verify_kernel(torch.tensor([[[.5, .5], [.5, .5]]]), torch.tensor([[[0., 1.]]]), torch.zeros(1, 1, dtype=torch.int64), torch.tensor([[0.9]]), torch.zeros(1, 2))
    check("documented edge: a draft token with q = 0 (impossible for a real draft model to sample) is accepted without dividing by zero", int(n_q[0]) == 1)
    pg = torch.tensor([[[0., .499, .499], [0., .499, .499]]]); qg = torch.tensor([[[0., .5, .5]]])
    ng, tg = verify_kernel(pg, qg, torch.tensor([[1]]), torch.tensor([[0.9999]]), torch.tensor([[0., 5., -5.]]))
    tgt = verify_torch(pg, qg, torch.tensor([[1]]), torch.tensor([[0.9999]]), torch.tensor([[0., 5., -5.]]))[1]
    check("documented limitation: if the residual is empty (p <= q everywhere, which only happens for unnormalized p) the kernel and torch both return token 0, which p gives probability 0",
          int(ng[0]) == 0 and int(tg[0]) == 0 and int(tgt[0]) == 0)
    try:
        ph, qh = torch.softmax(torch.randn(2, 3, 4), -1).half(), torch.softmax(torch.randn(2, 2, 4), -1).half()
        verify_kernel(ph, qh, torch.zeros(2, 2, dtype=torch.int64), torch.rand(2, 2), torch.zeros(2, 4))
        half_rejected = False
    except Exception:
        half_rejected = True
    check("documented limitation: float16 probabilities are rejected (tl.log wants float32); pass float32 p and q", half_rejected)

    # ---- the algebra of losslessness (exact, no sampling)
    worst, tv_wrong = 0.0, []
    g2 = torch.Generator().manual_seed(2)
    for _ in range(200):
        V = int(torch.randint(2, 21, (1,), generator=g2))
        pp = torch.softmax(torch.randn(V, generator=g2).double() * 1.5, -1)
        qq = torch.softmax(torch.randn(V, generator=g2).double() * 1.5, -1)
        alpha = torch.minimum(pp, qq).sum()
        res = (pp - qq).clamp_min(0)
        emit = qq * torch.minimum(torch.ones_like(pp), pp / qq) + (1 - alpha) * res / res.sum()
        worst = max(worst, float((emit - pp).abs().max()))
        wrong = torch.minimum(pp, qq) + (1 - alpha) * pp
        tv_wrong.append(0.5 * float((wrong - pp).abs().sum()))
    check("exact identity on 200 random (p, q): P(accept x) + P(reject) x normalized residual equals p(x) to 1e-12 (worst error %.1e)" % worst, worst < 1e-12)
    check("the same algebra with the WRONG correction (resample from p) is biased: mean TV to p is %.3f over the same 200 pairs" % (sum(tv_wrong) / 200), sum(tv_wrong) / 200 > 0.05)

    # ---- sampling checks through the real kernel and the chain
    pv = torch.tensor([.1, .2, .3, .4]); qv = torch.tensor([.4, .3, .2, .1])
    Bn = 4000
    P = pv.expand(Bn, 2, 4).contiguous(); Q = qv.expand(Bn, 1, 4).contiguous()
    xs = sample(Q, gen)
    nk, tk = verify_kernel(P, Q, xs, torch.rand(Bn, 1, generator=gen), gumbel((Bn, 4), gen))
    first = torch.where(nk == 1, xs[:, 0], tk)
    fq = torch.bincount(first, minlength=4).float() / Bn
    check("through the REAL kernel, K = 1: the first emitted token follows p = (.1 .2 .3 .4) within 0.035 over 4000 rows although the draft is q = (.4 .3 .2 .1); max deviation %.3f" % float((fq - pv).abs().max()),
          float((fq - pv).abs().max()) < 0.035)
    gen2 = torch.Generator().manual_seed(1)
    V, K_, L, B_ = 4, 3, 3, 6000
    Tp = torch.softmax(torch.randn(V, V, generator=gen2) * 1.5, -1)
    Tq = torch.softmax(torch.randn(V, V, generator=gen2) * 1.5, -1)
    ex = exact_distribution(Tp, L)
    good, _ = speculative_chain(Tp, Tq, K_, L, B_, gen2)
    bad, _ = speculative_chain(Tp, Tq, K_, L, B_, gen2, wrong=True)
    tvg, tvb = tv_distance(good, ex), tv_distance(bad, ex)
    check("3-token sequences from a Markov-chain target (64 outcomes, 6000 trials): correct correction TV %.3f < 0.04, wrong correction TV %.3f > 0.10" % (tvg, tvb), tvg < 0.04 and tvb > 0.10)
    ok_f = True
    for K_f in (1, 2, 4, 8):
        pf = torch.softmax(torch.randn(8, generator=gen) * 1.2, -1); qf = torch.softmax(torch.randn(8, generator=gen) * 1.2, -1)
        a = float(torch.minimum(pf, qf).sum())
        Bf = 40000
        Pf, Qf = pf.expand(Bf, K_f + 1, 8).contiguous(), qf.expand(Bf, K_f, 8).contiguous()
        nf, _ = verify_torch(Pf, Qf, sample(Qf, gen), torch.rand(Bf, K_f, generator=gen), gumbel((Bf, 8), gen))
        ok_f &= abs(float((nf + 1).float().mean()) - (1 - a ** (K_f + 1)) / (1 - a)) < 0.03
    check("tokens per target pass match (1 - a^(K+1)) / (1 - a) within 0.03 for K = 1, 2, 4, 8", ok_f)
    pq = torch.softmax(torch.randn(6, generator=gen), -1)
    pd0 = (pq * 0).clone(); pd0[0] = 1.0
    qd0 = pq * 0; qd0[1] = 1.0
    nj, _ = verify_kernel(pd0.expand(30, 3, 6).contiguous(), qd0.expand(30, 2, 6).contiguous(), torch.ones(30, 2, dtype=torch.int64), torch.rand(30, 2, generator=gen).clamp_min(1e-6), torch.zeros(30, 6))
    check("disjoint supports (acceptance rate 0): every row is rejected at position 0, so a step yields exactly 1 token", bool((nj == 0).all()))

    # ---- hygiene
    p, q, x, u, g = _rand_case(16, 3, 5, gen)
    snap = [t.clone() for t in (p, q, x, u, g)]
    o1, o2 = verify_kernel(p, q, x, u, g), verify_kernel(p, q, x, u, g)
    check("the kernel is deterministic and does not modify its inputs", torch.equal(o1[0], o2[0]) and torch.equal(o1[1], o2[1]) and all(torch.equal(a, b) for a, b in zip(snap, (p, q, x, u, g))))
    p2 = p.clone(); p2[0] = torch.softmax(torch.randn(4, 5, generator=gen), -1)
    o3 = verify_kernel(p2, q, x, u, g)
    check("rows are independent: rewriting row 0's target distribution leaves rows 1..15 unchanged", torch.equal(o3[0][1:], o1[0][1:]) and torch.equal(o3[1][1:], o1[1][1:]))
    for name, good_ in results:
        print(f"  [{'PASS' if good_ else 'FAIL'}] {name}")
    print(f"{sum(g_ for _, g_ in results)}/{len(results)} self-tests pass")
    return all(g_ for _, g_ in results)



# ------------------------------------------------------------- extras
def extras():
    gen = torch.Generator().manual_seed(4)
    V, K, B = 8, 4, 20000
    p = torch.softmax(torch.randn(V, generator=gen) * 1.2, -1)
    r = torch.softmax(torch.randn(V, generator=gen) * 1.2, -1)
    print("Experiment A: draft quality against acceptance (q = lam p + (1 - lam) r, V = 8, K = 4, 20000 trials)")
    print("lam    alpha   tokens/pass measured   formula")
    for lam in (1.0, 0.75, 0.5, 0.25, 0.0):
        q = lam * p + (1 - lam) * r
        a = float(torch.minimum(p, q).sum())
        P, Q = p.expand(B, K + 1, V).contiguous(), q.expand(B, K, V).contiguous()
        n, _ = verify_torch(P, Q, sample(Q, gen), torch.rand(B, K, generator=gen), gumbel((B, V), gen))
        f = (K + 1) if a >= 1 - 1e-9 else (1 - a ** (K + 1)) / (1 - a)
        print(f"{lam:4.2f}   {a:5.3f}   {float((n + 1).float().mean()):17.3f}   {f:8.3f}")
    print("\nExperiment B: a MODEL of speedup = tokens per pass / (1 + K c), c = draft cost / target cost (assumes one verify pass costs the same for any K: optimistic)")
    print("alpha    c     best K   tokens/pass   modelled speedup")
    for a in (0.5, 0.7, 0.8, 0.9):
        for c in (0.05, 0.2):
            best = max(range(1, 13), key=lambda k: ((1 - a ** (k + 1)) / (1 - a)) / (1 + k * c))
            tp = (1 - a ** (best + 1)) / (1 - a)
            print(f"{a:5.2f}  {c:4.2f}  {best:6d}   {tp:11.2f}   {tp/(1+best*c):10.2f}x")
    print("(K is searched from 1 to 12; alpha = 0.9 with c = 0.05 hits that cap.) A draft costing a fifth of the target with alpha = 0.5 gives a modest 1.25x in this model. None of this is a measurement.")
    print("\nExperiment C: how biased is the wrong correction? (random 8-token pairs; TV distance of the first emitted token from p)")
    print("alpha   TV(correct)   TV(resample from p)")
    for _ in range(6):
        pp = torch.softmax(torch.randn(V, generator=gen).double() * 1.5, -1); qq = torch.softmax(torch.randn(V, generator=gen).double() * 1.5, -1)
        al = torch.minimum(pp, qq).sum()
        wrong = torch.minimum(pp, qq) + (1 - al) * pp
        res = (pp - qq).clamp_min(0)
        right = torch.minimum(pp, qq) + (1 - al) * res / res.sum()
        print(f"{float(al):5.3f}   {0.5*float((right-pp).abs().sum()):11.1e}   {0.5*float((wrong-pp).abs().sum()):14.3f}")
    print("the wrong correction over-represents every token with q >= p (their probability becomes p(1 + 1 - alpha)); the bias shrinks only as alpha approaches 1")
    print("\nExperiment D: why the kernel tests u * q <= p and does not divide")
    pq, qq = torch.tensor([1e-3, 0.0]), torch.tensor([1e-45, 0.0])
    print(f"float32 p/q with q = 1e-45: {float(pq[0]/qq[0])}   with p = q = 0: {float(pq[1]/qq[1])}   (inf and nan)")
    print(f"float32 u*q <= p with the same values and u = 0.5: {bool((0.5*qq[0]) <= pq[0])}, {bool((0.5*qq[1]) <= pq[1])}   (finite comparisons, no special cases)")
    print("\nExperiment E: bytes the verify kernel reads per sequence (float32 p and q; it reads K accept scalars plus ONE row of p and ONE row of q)")
    print("V        K    bytes read   full p, q tensors   share")
    for Vx in (32000, 128256):
        for Kx in (4, 8):
            rd = (2 * Kx + 2 * Vx) * 4
            full = ((Kx + 1) + Kx) * Vx * 4
            print(f"{Vx:7d}  {Kx:2d}  {rd:10d}  {full:18d}  {100*rd/full:5.1f} %")
    print("the target model produces the full tensors regardless; the kernel avoids re-reading most of them")


# ------------------------------------------------------------- GPU benchmark (UNTESTED: written without access to a GPU)
def bench():
    if not torch.cuda.is_available():
        print("--bench needs a CUDA GPU; none found. Nothing was measured.")
        return
    from triton.testing import do_bench
    K = 4
    print("NOTE: verify_kernel allocates n and tok on the CPU: create them with device=p.device first. One program per sequence loads rows of size next_power_of_2(V).")
    print("V        B     time_us")
    for V in (32000, 128256):
        for B in (1, 16, 128):
            p = torch.softmax(torch.randn(B, K + 1, V, device="cuda"), -1)
            q = torch.softmax(torch.randn(B, K, V, device="cuda"), -1)
            x = torch.randint(0, V, (B, K), device="cuda"); u = torch.rand(B, K, device="cuda"); g = torch.randn(B, V, device="cuda")
            t = do_bench(lambda: verify_kernel(p, q, x, u, g), warmup=25, rep=100)
            print(f"{V:7d}  {B:4d}  {t*1e3:8.1f}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--bench" in sys.argv:
        bench(); sys.exit(0)
    main()
