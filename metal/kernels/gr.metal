// metal/kernels/gr.metal - the hyper-connection (gated residual) read for a window of tokens, with the previous
// sublayer's write folded in.  The math is fused_gr.hpp's (native contract: FP32 activations, BF16 weights):
//
//   gr_norm  R'[c] = R[c] + bo * 2 sigmoid(inj_prev[c] / hc)       (when apply; written back in place)
//            xn[c] = R'[c] * w_norm[c] * rsqrt(mean(R'[c]^2) + eps)
//   gr_down  lo[k] = silu((w_down[k] . xn) / hc)   k < hc_lr;   inject[c] = w_inject[c] . xn
//   gr_up    mixed[d] = mean_c xn[c][d] * sigmoid(w_up[c n + d] . lo), written as prepared activations (xprep's
//            planes and per-16 sums) for the GEMVs that read it
//
// Shapes: n_embd 2560, hc 4, hc_lr 320.  Per token t: R[t] (hc * n), bo[t] (n), inj_prev[t] (hc), xn[t] (hc * n),
// lo[t] (hc_lr), inject[t] (hc), XP planes of n_tok rows.

constant int GR_NTOK [[function_constant(1)]];

struct GrArgs {
    uint n_embd;
    uint hc;
    uint hc_lr;
    uint n_tok;
    uint apply;
    float eps;
    uint nsplit;   // gr_norm_p1/p2's slice count over one stream's row (others ignore it)
};

// one threadgroup per (stream c, token t), 256 threads
kernel void gr_norm(device float* R [[buffer(0)]], device const float* bo [[buffer(1)]],
                    device const float* inj [[buffer(2)]], device const float* w_norm [[buffer(3)]],
                    device float* xn [[buffer(4)]], constant GrArgs& a [[buffer(5)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    device float* r = R + (t * a.hc + c) * n;
    const float g = a.apply ? 2.0f / (1.0f + exp(-inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        float v = r[d];
        if (a.apply) {
            v = fma(bo[t * n + d], g, v);
            r[d] = v;
        }
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = rsqrt(tot / float(n) + a.eps);
    device float* o = xn + (t * a.hc + c) * n;
    device const float* w = w_norm + c * n;
    for (uint d = tid; d < n; d += 256) o[d] = r[d] * w[d] * rs;
}

// the split hyper-connection norm: gr_norm in two phases is only worthwhile at small n_tok (decode), where the
// one-threadgroup-per-(stream, token) original runs on 4 threadgroups.  Phase 1 (grid (hc * 8, n_tok), 256
// threads) folds the gate and w_norm in and atomically adds each slice's fp32 partial sum of squares at
// partial[t * hc + c]; phase 2 (grid (hc, n_tok)) reduces the slices into rsbuf[t * hc + c] and zeroes the
// partials for the next call.  The consumers then scale xn's slices by rs themselves (gr_down per unit,
// gr_up per stream), which preserves the original rounding order of o = (R * w) * rs.
// gr_norm in two phases: phase 1 (grid (hc * ns, n_tok), 256 threads) folds the gate and w_norm in and atomically
// adds each slice's fp32 partial sum of squares at partial[t * hc + c]; phase 2 (grid (hc, n_tok)) reduces the
// slices into rsbuf[t * hc + c] and zeroes the partials for the next call.  ns = a.nsplit (runtime-swappable).
kernel void gr_norm_p1(device float* R [[buffer(0)]], device const float* bo [[buffer(1)]],
                       device const float* inj [[buffer(2)]], device const float* w_norm [[buffer(3)]],
                       device float* xn [[buffer(4)]], constant GrArgs& a [[buffer(5)]],
                       device float* partial [[buffer(6)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint ns = a.nsplit ? a.nsplit : 8u;
    const uint c = tg.x / ns, t = tg.y;
    const uint n = a.n_embd, n0 = (n + ns - 1) / ns;   // slice width (ceil: any ns works)
    const uint d0 = (tg.x % ns) * n0, d1 = min(d0 + n0, n);
    device float* r = R + (t * a.hc + c) * n + d0;
    device float* o = xn + (t * a.hc + c) * n + d0;
    device const float* wn = w_norm + c * n + d0;
    const float g = a.apply ? 2.0f / (1.0f + exp(-inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    float ss = 0.0f;
    for (uint d = tid; d < d1 - d0; d += 256) {
        float v = r[d];
        if (a.apply) {
            v = fma(bo[t * n + d0 + d], g, v);
            r[d] = v;
        }
        ss = fma(v, v, ss);
        o[d] = v * wn[d];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += part[i];
        atomic_fetch_add_explicit((device atomic_float*)(partial + t * a.hc + c), tot, memory_order_relaxed);
    }
}
kernel void gr_norm_p2(device float* partial [[buffer(0)]], device float* rsbuf [[buffer(1)]],
                       constant GrArgs& a [[buffer(2)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint tc = tg.y * a.hc + tg.x;
    if (tid == 0) {
        // the split partials were atomically added into this one slot by gr_norm_p1: read it once, then zero it
        const float tot = partial[tc];
        rsbuf[tc] = rsqrt(tot / float(a.n_embd) + a.eps);
        partial[tc] = 0.0f;
    }
}

// rows: 0 .. hc_lr - 1 = w_down, hc_lr .. hc_lr + hc - 1 = w_inject (both bf16 rows of hc * n_embd).
// SG simdgroups split the row's units KS ways; LPR = 32 (a simdgroup per row); RS = SG / KS rows per threadgroup.
// one row's K-split partial over xn (units of format F).  rsv (rsbuf, indexed t * hc + stream) applies the split
// norm's rs here, per unit (a unit never spans two streams: n_embd / UNIT units a stream), preserving the
// original rounding order of the folded gr_norm ((R * w) first, then * rs).
template <int F>
static inline void grd_acc(device const uchar* w, int nin, int ks, int KS, ushort lane, device const float* xn,
                           device const float* rsv, int hc, int n_embd, thread float* acc, threadgroup const uint* notab) {
    constexpr int UNIT = fmt<F>::UNIT;
    const int U = nin / UNIT, u0 = ks * U / KS, u1 = (ks + 1) * U / KS;
    for (int u = u0 + lane; u < u1; u += 32) {
        const int rc = (uint)u * UNIT / n_embd;   // this unit's stream (a unit never spans two streams)
        float q[UNIT];
        float4 sc;
        Dec<F>::run(w, u, q, sc, notab);
        for (int t = 0; t < GR_NTOK; ++t) {
            const float rs = rsv[(uint)(t * hc + rc)];
            float xs[UNIT], xsum[2] = {0.0f, 0.0f};
            for (int k = 0; k < UNIT; k += 16) load16(xn + t * nin + UNIT * u + k, xs + k);
            for (int k = 0; k < UNIT; ++k) xs[k] *= rs;
            acc[t] += unit_combine<UNIT>(q, sc, xs, xsum);
        }
    }
}

// FI: the inject rows' format (the Unsloth files: Q8_0 down, F32 inject); FI = F is the original kernel
template <int SG, int KS, int F, int FI>
kernel void gr_down_f(device const uchar* w_down [[buffer(0)]], device const uchar* w_inject [[buffer(1)]],
                    device const float* xn [[buffer(2)]], device float* lo [[buffer(3)]],
                    device float* inject [[buffer(4)]], device const float* rsv [[buffer(5)]],
                    constant GrArgs& a [[buffer(6)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int RS = SG / KS;
    threadgroup float red[KS * RS * 8];
    threadgroup uint notab[1];
    const int ks = sg / RS, rs = sg % RS;
    const int row = int(tg.x) * RS + rs;
    const int nrows = int(a.hc_lr + a.hc);
    const int rr = min(row, nrows - 1);
    const int nin = int(a.hc * a.n_embd);
    const uint rb = uint(nin / fmt<F>::QK * fmt<F>::BB), rbi = uint(nin / fmt<FI>::QK * fmt<FI>::BB);
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    if (rr < int(a.hc_lr)) grd_acc<F>(w_down + ulong(rr) * rb, nin, ks, KS, lane, xn, rsv, int(a.hc), int(a.n_embd), acc, notab);
    else grd_acc<FI>(w_inject + ulong(rr - int(a.hc_lr)) * rbi, nin, ks, KS, lane, xn, rsv, int(a.hc), int(a.n_embd), acc, notab);
    for (int t = 0; t < GR_NTOK; ++t) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) red[(ks * RS + rs) * 8 + t] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0 && lane < GR_NTOK && row < nrows) {
        const int t = lane;
        float v = 0.0f;
        for (int k = 0; k < KS; ++k) v += red[(k * RS + rs) * 8 + t];
        if (row < int(a.hc_lr)) {
            const float z = v / float(a.hc);
            lo[t * int(a.hc_lr) + row] = z / (1.0f + exp(-z));
        } else {
            inject[t * int(a.hc) + row - int(a.hc_lr)] = v;
        }
    }
}

// A threadgroup owns 16 consecutive d for all hc = 4 streams: 8 simdgroups, simdgroup s -> stream s / 2 and
// d0 + 8 (s % 2) .. + 8; LPR = 4 lanes per row (320 = 20 units of 16, 5 per lane).
// XP: [4 kinds][n_tok][n] then S16 [n_tok][n / 16] (units.metal's xprep layout).
// a threadgroup covers 16 consecutive d for each stream; LPR lanes of a simdgroup share one stream-row
// (LPR=4 is the original layout: dl = 8 * (sg & 1) + lane / 4); instantiated as LPR 4 only (2 measured slower).
template <int F, int LPR>
kernel void gr_up_f(device const uchar* w_up [[buffer(0)]], device const float* lo [[buffer(1)]],
                  device const float* xn [[buffer(2)]], device float* xp [[buffer(3)]],
                  device const float* rsv [[buffer(4)]], constant GrArgs& a [[buffer(5)]],
                  uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float gate[8][4][16];
    threadgroup float msum[8][16];
    threadgroup uint notab[1];
    const int n = int(a.n_embd), c = sg >> 1, dl = 8 * (sg & 1) + lane / LPR, li = lane % LPR;
    const int d = int(tg.x) * 16 + dl;
    constexpr int UNIT = fmt<F>::UNIT;
    device const uchar* w = w_up + ulong(c * n + d) * (a.hc_lr / fmt<F>::QK * fmt<F>::BB);
    const int U = int(a.hc_lr) / UNIT;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int u = li; u < U; u += LPR) {
        float q[UNIT];
        float4 sc;
        Dec<F>::run(w, u, q, sc, notab);
        for (int t = 0; t < GR_NTOK; ++t) {
            float xs[UNIT], xsum[2] = {0.0f, 0.0f};
            for (int k = 0; k < UNIT; k += 16) load16(lo + t * int(a.hc_lr) + UNIT * u + k, xs + k);
            acc[t] += unit_combine<UNIT>(q, sc, xs, xsum);
        }
    }
    for (int t = 0; t < GR_NTOK; ++t) {
        const float v = lane_sum<LPR>(acc[t]);
        if (li == 0) gate[t][c][dl] = 1.0f / (1.0f + exp(-v));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 16 threads per token: thread j -> d0 + j
    const int t = tid / 16, j = tid & 15;
    if (t < GR_NTOK) {
        const int dd = int(tg.x) * 16 + j;
        const int hc = int(a.hc);
        float m = 0.0f;
        for (int cc = 0; cc < hc; ++cc) m = fma(xn[(t * hc + cc) * n + dd], rsv[t * hc + cc] * gate[t][cc][j], m);
        m = m / float(hc);
        const uint plane = a.n_tok * uint(n);
        device float* raw = xp + t * n + dd;
        const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f,
                              0.00006103515625f};
        const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
        raw[0] = m;
        raw[plane] = m * q2s[j & 7];
        raw[2 * plane] = m * p4s[j & 3];
        raw[3 * plane] = (j & 1) ? m * 0.00390625f : m;
        msum[t][j] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < GR_NTOK && j == 0) {
        // the per-16 sum in xprep's order (even and odd partial sums, then added)
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += msum[t][k]; s1 += msum[t][k + 1]; }
        xp[4 * a.n_tok * uint(n) + t * (n / 16) + tg.x] = s0 + s1;
    }
}

#define GRD(S, K) template [[host_name("gr_down_" #S "_" #K)]] kernel decltype(gr_down_f<S, K, T_BF16, T_BF16>) gr_down_f<S, K, T_BF16, T_BF16>; \
    template [[host_name("gr_down_q8_" #S "_" #K)]] kernel decltype(gr_down_f<S, K, T_Q8_0, T_Q8_0>) gr_down_f<S, K, T_Q8_0, T_Q8_0>; \
    template [[host_name("gr_down_q8f_" #S "_" #K)]] kernel decltype(gr_down_f<S, K, T_Q8_0, T_F32>) gr_down_f<S, K, T_Q8_0, T_F32>;
GRD(8, 8) GRD(8, 4) GRD(4, 4) GRD(4, 2) GRD(16, 16) GRD(16, 8)
template [[host_name("gr_up")]] kernel decltype(gr_up_f<T_BF16, 4>) gr_up_f<T_BF16, 4>;
template [[host_name("gr_up_q8")]] kernel decltype(gr_up_f<T_Q8_0, 4>) gr_up_f<T_Q8_0, 4>;
