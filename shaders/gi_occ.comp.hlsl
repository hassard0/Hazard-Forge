// Slice GI5 — Deterministic Lumen-class GI: the INTEGER CHEBYSHEV OCCLUSION-WEIGHTED interpolation compute
// pass (the probe-volume LIGHT-LEAK FIX of FLAGSHIP #29: DETERMINISTIC LUMEN-CLASS GI). ONE thread per QUERY
// POINT (i < queryCount). Each thread reads its query point + normal, runs gi::FxInterpolateIrradianceOcc
// VERBATIM — the GI3 trilinear blend (gi_interp.comp) but each of the 8 corner weights is scaled by the
// variance-shadow CHEBYSHEV VISIBILITY of that corner probe (from the per-probe distance moments gMoments)
// and the 8 weights are RE-NORMALIZED to Σ == kOne — then FxInterpolateSH + FxSHEvaluate. Writes the
// GiRadiance to gOut[i]. Bit-identical to the CPU gi::InterpolateFieldOcc / FxInterpolateIrradianceOcc.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it consumes
// the host-snapped Q16.16 query points/normals + the GI2 integer SH buffer + the GI5 integer moments buffer
// (gi::FxProbeMoments, computed by the CPU gi::FxProbeMoments_All from the RT trace t and uploaded — the
// moments are a cheap by-product, no GPU distance pass) and runs the SAME pure-integer math the header runs:
// the GI3 partition-of-unity trilinear weights, the per-corner Chebyshev fxdiv visibility, the lerp(kOne,
// vis, occStrength) factor, the fxdiv re-normalize (last corner = exact leftover so Σ == kOne), the int64
// 8-corner SH blend (narrow ONCE >> kFrac), and the int64 cosine-lobe reconstruction (narrow ONCE >>
// 2*kFrac). A divergence vs engine/render/gi.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): the SH blend term weight(Q16.16) * coeff(Q16.16) is a Q32.32 product
// (overflowing int32), the cosine-lobe reconstruction is a triple Q16.16 product (-> Q48.48), AND the t²
// moments / Chebyshev denominator are int64 — so they MUST be held in int64 and quantized once. HLSL SM6
// supports int64_t (the gi_interp.comp / gi_sh_encode.comp pattern — DXC -spirv with the Int64 capability).
// glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is
// VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --gi5-occlusion
// showcase runs the CPU gi::InterpolateFieldOcc (byte-identical by construction, the gi_interp.comp
// convention).
//
// NO RayQuery, NO accel structure — pure-integer compute over the GI2 SH buffer + the GI5 moments buffer
// (only the int64 width keeps the shader Vulkan-only; the math is otherwise strict-zero on the Vulkan side).
//
// Buffers (storage, compute bindings 0..5):
//   b0 gSH      : the GI2 FxProbeSH array (probeCount; 9*3 coeffs + pad == 28 ints == 112 B), READ.
//   b1 gPoints  : the query points (queryCount; x,y,z,_pad Q16.16 int4 == 16 B), READ.
//   b2 gNormals : the query normals (queryCount; x,y,z,_pad Q16.16 int4 == 16 B), READ.
//   b3 gOut     : the GiRadiance output (queryCount; r,g,b,_pad int4 == 16 B), WRITE.
//   b4 gParams  : { queryCount, probeCount, nx, ny | nz, ox, oy, oz | spacing, occStrength }, READ.
//   b5 gMoments : the GI5 FxProbeMoments array (probeCount; meanDist, meanDist2 == 2 ints == 8 B), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// gi_interp.comp), not backend CODE symbols.

#define HF_GI_THREADS 64
#define HF_GI_FRAC 16          // MUST match gi.h::kFrac
#define HF_GI_ONE  65536       // MUST match gi.h::kOne (1.0 in Q16.16)

// std430 FxProbeSH mirror (gi.h::FxProbeSH): 9*3 = 27 Q16.16 coeff ints + 1 pad int = 28 ints (112 B).
struct FxProbeSH {
    int c[28];
};

// std430 int4 mirror (gi.h::FxVec3 padded / GiRadiance): x,y,z,_pad Q16.16 ints, 16 bytes.
struct Vec4i {
    int x, y, z, w;
};

// std430 FxProbeMoments mirror (gi.h::FxProbeMoments): meanDist, meanDist2 (Q16.16) == 2 ints (8 B).
struct FxProbeMoments {
    int meanDist;
    int meanDist2;
};

// Params: cfg0 = { queryCount, probeCount, nx, ny }; cfg1 = { nz, originX, originY, originZ };
// cfg2 = { spacing, occStrength, 0, 0 }.
struct GiOccParams {
    int4 cfg0;   // x=queryCount, y=probeCount, z=nx, w=ny
    int4 cfg1;   // x=nz, y=originX, z=originY, w=originZ
    int4 cfg2;   // x=spacing, y=occStrength
};

[[vk::binding(0, 0)]] StructuredBuffer<FxProbeSH>      gSH      : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<Vec4i>          gPoints  : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<Vec4i>          gNormals : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Vec4i>        gOut     : register(u0);
[[vk::binding(4, 0)]] StructuredBuffer<GiOccParams>    gParams  : register(t3);
[[vk::binding(5, 0)]] StructuredBuffer<FxProbeMoments> gMoments : register(t4);

// The cosine-lobe reconstruction band factors B_l = 4*A_l (Q16.16), IDENTICAL to gi.h::kGiSHReconB*.
static const int64_t kReconB0 = 823550;   // 4*pi    in Q16.16
static const int64_t kReconB1 = 549033;   // 8*pi/3  in Q16.16
static const int64_t kReconB2 = 205887;   // pi      in Q16.16

// The GI5 integer variance floor (gi.h::kGiVarFloor == kOne/4096 == 16).
static const int kVarFloor = HF_GI_ONE / 4096;

// fxmul — Q16.16 multiply (int64 intermediate, narrow >> kFrac), IDENTICAL to gi.h fxmul.
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_GI_FRAC);
}
// fxdiv — Q16.16 divide ((a<<kFrac)/b), IDENTICAL to rtrace::fxdiv.
int fxdiv(int a, int b) {
    return (int)((((int64_t)a) << HF_GI_FRAC) / (int64_t)b);
}

// FxISqrt — integer sqrt of an int64 Q32.32 sum-of-squares -> Q16.16 length, IDENTICAL to fpx::FxISqrt
// (the binary-digit loop). Returns floor(sqrt(v)) treating v as Q32.32 (root is Q16.16).
int FxISqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t bit = (int64_t)1 << 62;
    while (bit > v) bit >>= 2;
    int64_t res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return (int)res;
}

// FxChebyshevVisibility — the integer variance-shadow visibility, IDENTICAL to gi.h::FxChebyshevVisibility.
int FxChebyshevVisibility(FxProbeMoments m, int queryDist) {
    if (queryDist <= m.meanDist) return HF_GI_ONE;
    int variance = m.meanDist2 - fxmul(m.meanDist, m.meanDist);
    if (variance < kVarFloor) variance = kVarFloor;
    int dd = queryDist - m.meanDist;
    int dd2 = fxmul(dd, dd);
    int denom = variance + dd2;
    int vis = fxdiv(variance, denom);
    if (vis < 0) vis = 0;
    if (vis > HF_GI_ONE) vis = HF_GI_ONE;
    return vis;
}

// GiFxLerp — a + (b-a)*t, IDENTICAL to gi.h::GiFxLerp. t==0 -> a EXACTLY.
int GiFxLerp(int a, int b, int t) {
    return a + fxmul(b - a, t);
}

// ProbePos integer decode (gi.h::ProbePos by linear index): cx-major decode + origin + (ix,iy,iz)*spacing.
void ProbePosI(int linearIndex, int nx, int ny, int ox, int oy, int oz, int spacing,
               out int px, out int py, out int pz) {
    int ix = linearIndex % nx;
    int iy = (linearIndex / nx) % ny;
    int iz = linearIndex / (nx * ny);
    px = ox + fxmul(ix * HF_GI_ONE, spacing);
    py = oy + fxmul(iy * HF_GI_ONE, spacing);
    pz = oz + fxmul(iz * HF_GI_ONE, spacing);
}

[numthreads(HF_GI_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int queryCount = gParams[0].cfg0.x;
    int probeCount = gParams[0].cfg0.y;
    int i = (int)gid.x;
    if (i >= queryCount) return;

    int nx = gParams[0].cfg0.z, ny = gParams[0].cfg0.w;
    int nz = gParams[0].cfg1.x;
    int ox = gParams[0].cfg1.y, oy = gParams[0].cfg1.z, oz = gParams[0].cfg1.w;
    int spacing = gParams[0].cfg2.x;
    int occStrength = gParams[0].cfg2.y;

    Vec4i pt = gPoints[i];
    Vec4i nm = gNormals[i];

    // ===== FxNearestProbes — floor-cell + frac + 8-corner index + partition-of-unity weights =====
    int base[3]; int frac[3];
    int pv[3] = { pt.x, pt.y, pt.z };
    int orig[3] = { ox, oy, oz };
    int dim[3] = { nx, ny, nz };
    [unroll] for (int ax = 0; ax < 3; ++ax) {
        if (dim[ax] <= 1) { base[ax] = 0; frac[ax] = 0; }
        else {
            int g = (int)((((int64_t)(pv[ax] - orig[ax])) << HF_GI_FRAC) / (int64_t)spacing);
            int b = g >> HF_GI_FRAC;
            if (b < 0) b = 0;
            if (b > dim[ax] - 2) b = dim[ax] - 2;
            int fr = (int)((int64_t)g - ((int64_t)b << HF_GI_FRAC));
            if (fr < 0) fr = 0;
            if (fr > HF_GI_ONE) fr = HF_GI_ONE;
            base[ax] = b; frac[ax] = fr;
        }
    }
    int wlo[3] = { HF_GI_ONE - frac[0], HF_GI_ONE - frac[1], HF_GI_ONE - frac[2] };
    int whi[3] = { frac[0], frac[1], frac[2] };

    int idx[8]; int wgt[8];
    int accumW = 0;
    [unroll] for (int c = 0; c < 8; ++c) {
        int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
        int cx = base[0] + sx; if (cx > nx - 1) cx = nx - 1;
        int cy = base[1] + sy; if (cy > ny - 1) cy = ny - 1;
        int cz = base[2] + sz; if (cz > nz - 1) cz = nz - 1;
        idx[c] = cx + cy * nx + cz * (nx * ny);
        int wx = sx ? whi[0] : wlo[0];
        int wy = sy ? whi[1] : wlo[1];
        int wz = sz ? whi[2] : wlo[2];
        if (c < 7) {
            int64_t p = (int64_t)wx * (int64_t)wy * (int64_t)wz;
            int wc = (int)(p >> (2 * HF_GI_FRAC));
            wgt[c] = wc;
            accumW += wc;
        } else {
            wgt[c] = HF_GI_ONE - accumW;
        }
    }

    // ===== GI5 — Chebyshev occlusion weighting + re-normalize (FxInterpolateIrradianceOcc) =====
    // Scale each corner weight by lerp(kOne, FxChebyshevVisibility(moments[corner], distToCorner),
    // occStrength); occStrength==0 -> factor kOne -> unchanged. Then re-normalize so Σ == kOne EXACTLY.
    bool haveMoments = (probeCount > 0);
    int scaled[8];
    int64_t sumW = 0;
    [unroll] for (int c = 0; c < 8; ++c) {
        int w = wgt[c];
        if (haveMoments && occStrength != 0) {
            int cpx, cpy, cpz;
            ProbePosI(idx[c], nx, ny, ox, oy, oz, spacing, cpx, cpy, cpz);
            int dx = pt.x - cpx;
            int dy = pt.y - cpy;
            int dz = pt.z - cpz;
            int64_t ss = (int64_t)dx * (int64_t)dx + (int64_t)dy * (int64_t)dy + (int64_t)dz * (int64_t)dz;
            int distToCorner = FxISqrt(ss);
            int vis = FxChebyshevVisibility(gMoments[idx[c]], distToCorner);
            int factor = GiFxLerp(HF_GI_ONE, vis, occStrength);
            w = fxmul(w, factor);
        }
        scaled[c] = w;
        sumW += (int64_t)w;
    }
    // Re-normalize (last corner = exact leftover). sumW==kOne (occStrength==0) -> fxdiv(w,kOne)==w -> identity.
    int normW[8];
    [unroll] for (int c = 0; c < 8; ++c) normW[c] = wgt[c];
    if (sumW > 0) {
        int accum = 0;
        [unroll] for (int c = 0; c < 7; ++c) {
            int w = fxdiv(scaled[c], (int)sumW);
            normW[c] = w;
            accum += w;
        }
        normW[7] = HF_GI_ONE - accum;
    }

    // ===== FxInterpolateSH — int64 8-corner blend, narrow ONCE =====
    int blended[27];
    if (probeCount <= 0) {
        [unroll] for (int z0 = 0; z0 < 27; ++z0) blended[z0] = 0;
    } else {
        [unroll] for (int k = 0; k < 27; ++k) {
            int64_t acc = 0;
            [unroll] for (int c2 = 0; c2 < 8; ++c2)
                acc += (int64_t)normW[c2] * (int64_t)gSH[idx[c2]].c[k];
            blended[k] = (int)(acc >> HF_GI_FRAC);
        }
    }

    // ===== FxSHEvaluate — the integer cosine-lobe irradiance reconstruction =====
    int x = nm.x, y = nm.y, z = nm.z;
    const int kY00f = 18487, kY1f = 32021, kY2af = 71601, kY2bf = 20670, kY2cf = 35801;
    int Y[9];
    Y[0] = kY00f;
    Y[1] = fxmul(kY1f, y);
    Y[2] = fxmul(kY1f, z);
    Y[3] = fxmul(kY1f, x);
    Y[4] = fxmul(kY2af, fxmul(x, y));
    Y[5] = fxmul(kY2af, fxmul(y, z));
    Y[6] = fxmul(kY2bf, fxmul(fxmul(3 * HF_GI_ONE, z), z) - HF_GI_ONE);
    Y[7] = fxmul(kY2af, fxmul(x, z));
    Y[8] = fxmul(kY2cf, fxmul(x, x) - fxmul(y, y));

    int64_t B[9] = { kReconB0, kReconB1, kReconB1, kReconB1,
                     kReconB2, kReconB2, kReconB2, kReconB2, kReconB2 };

    int outc[3];
    [unroll] for (int ch = 0; ch < 3; ++ch) {
        int64_t acc = 0;
        [unroll] for (int j = 0; j < 9; ++j)
            acc += (int64_t)blended[j * 3 + ch] * (int64_t)Y[j] * B[j];
        int v = (int)(acc >> (2 * HF_GI_FRAC));
        outc[ch] = (v < 0) ? 0 : v;
    }

    Vec4i res;
    res.x = outc[0]; res.y = outc[1]; res.z = outc[2]; res.w = 0;
    gOut[i] = res;
}
