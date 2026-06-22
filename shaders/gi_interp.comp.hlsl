// Slice GI3 — Deterministic Lumen-class GI: the INTEGER TRILINEAR SH INTERPOLATION compute pass (the
// continuous irradiance field of FLAGSHIP #29: DETERMINISTIC LUMEN-CLASS GI). ONE thread per QUERY POINT
// (i < queryCount). Each thread reads its query point + normal, runs FxInterpolateIrradiance VERBATIM —
// FxSHEvaluate(FxInterpolateSH(FxNearestProbes(grid, point), gSH), normal) — and writes the GiRadiance to
// gOut[i]. Bit-identical to the CPU gi::InterpolateField / FxInterpolateIrradiance.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it consumes
// the host-snapped Q16.16 query points/normals + the GI2 integer SH buffer and runs the SAME pure-integer
// math the header runs: the partition-of-unity trilinear weights (each axis wlo+whi == kOne EXACTLY, the
// 8-corner triple products narrowed >> 2*kFrac with the LAST corner taking the exact leftover so Σw == kOne),
// the int64 8-corner SH blend (narrow ONCE >> kFrac), and the int64 cosine-lobe reconstruction (narrow ONCE
// >> 2*kFrac). A divergence vs engine/render/gi.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): the SH blend term weight(Q16.16) * coeff(Q16.16) is a Q32.32 product
// (~1.0*~1.0 = 65536^2 = 4.29e9 > 2^31, overflowing int32), and the cosine-lobe reconstruction is a triple
// Q16.16 product (coeff*basis*B_l -> Q48.48), so both MUST be held in int64 and quantized ONCE. HLSL SM6
// supports int64_t (the gi_sh_encode.comp / grain_integrate.comp / fluid_integrate.comp pattern — DXC -spirv
// with the Int64 capability). glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// this shader is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the
// --gi3-interp showcase runs the CPU gi::InterpolateField (byte-identical by construction).
//
// NO RayQuery, NO accel structure — pure-integer compute over the GI2 SH buffer (only the int64 width keeps
// the shader Vulkan-only; the math is otherwise an MSL-native-class strict-zero proof on the Vulkan side).
//
// Buffers (storage, compute bindings 0..3):
//   b0 gSH      : the GI2 FxProbeSH array (probeCount; 9*3 coeffs + pad == 28 ints == 112 B), READ.
//   b1 gPoints  : the query points (queryCount; x,y,z,_pad Q16.16 int4 == 16 B), READ.
//   b2 gNormals : the query normals (queryCount; x,y,z,_pad Q16.16 int4 == 16 B), READ.
//   b3 gOut     : the GiRadiance output (queryCount; r,g,b,_pad int4 == 16 B), WRITE.
//   b4 gParams  : { queryCount, probeCount, nx | ny<<16, nz, originX, originY, originZ, spacing }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// gi_sh_encode.comp), not backend CODE symbols.

#define HF_GI_THREADS 64
#define HF_GI_FRAC 16          // MUST match gi.h::kFrac
#define HF_GI_ONE  65536       // MUST match gi.h::kOne (1.0 in Q16.16)

// std430 FxProbeSH mirror (gi.h::FxProbeSH): 9*3 = 27 Q16.16 coeff ints + 1 pad int = 28 ints (112 B).
// c[i*3 + ch] == coeff[basis i][channel ch]; c[27] == pad.
struct FxProbeSH {
    int c[28];
};

// std430 int4 mirror (gi.h::FxVec3 padded / GiRadiance): x,y,z,_pad Q16.16 ints, 16 bytes.
struct Vec4i {
    int x, y, z, w;
};

// Params: cfg0 = { queryCount, probeCount, nx, ny }; cfg1 = { nz, originX, originY, originZ };
// cfg2 = { spacing, 0, 0, 0 }.
struct GiInterpParams {
    int4 cfg0;   // x=queryCount, y=probeCount, z=nx, w=ny
    int4 cfg1;   // x=nz, y=originX, z=originY, w=originZ
    int4 cfg2;   // x=spacing
};

[[vk::binding(0, 0)]] StructuredBuffer<FxProbeSH>      gSH      : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<Vec4i>          gPoints  : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<Vec4i>          gNormals : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Vec4i>        gOut     : register(u0);
[[vk::binding(4, 0)]] StructuredBuffer<GiInterpParams> gParams  : register(t3);

// The cosine-lobe reconstruction band factors B_l = 4*A_l (Q16.16), IDENTICAL to gi.h::kGiSHReconB*.
static const int64_t kReconB0 = 823550;   // 4*pi    in Q16.16
static const int64_t kReconB1 = 549033;   // 8*pi/3  in Q16.16
static const int64_t kReconB2 = 205887;   // pi      in Q16.16

// fxmul — Q16.16 multiply (int64 intermediate, narrow >> kFrac), IDENTICAL to sim/fpx.h::fxmul.
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_GI_FRAC);
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

    Vec4i pt = gPoints[i];
    Vec4i nm = gNormals[i];

    // ===== FxNearestProbes — floor-cell + frac + 8-corner index + partition-of-unity weights =====
    // Per axis: g = (v - o)/spacing (Q16.16); base = floor(g) = g >> kFrac (arithmetic shift), clamped to
    // [0, dim-2]; frac = g - (base<<kFrac), clamped to [0, kOne]. dim==1 -> base 0, frac 0.
    int base[3]; int frac[3];
    int pv[3] = { pt.x, pt.y, pt.z };
    int orig[3] = { ox, oy, oz };
    int dim[3] = { nx, ny, nz };
    [unroll] for (int ax = 0; ax < 3; ++ax) {
        if (dim[ax] <= 1) { base[ax] = 0; frac[ax] = 0; }
        else {
            int g = (int)((((int64_t)(pv[ax] - orig[ax])) << HF_GI_FRAC) / (int64_t)spacing);
            int b = g >> HF_GI_FRAC;                       // floor toward -inf
            if (b < 0) b = 0;
            if (b > dim[ax] - 2) b = dim[ax] - 2;
            int fr = (int)((int64_t)g - ((int64_t)b << HF_GI_FRAC));
            if (fr < 0) fr = 0;
            if (fr > HF_GI_ONE) fr = HF_GI_ONE;
            base[ax] = b; frac[ax] = fr;
        }
    }
    // Per-axis lo/hi weights (wlo + whi == kOne EXACTLY).
    int wlo[3] = { HF_GI_ONE - frac[0], HF_GI_ONE - frac[1], HF_GI_ONE - frac[2] };
    int whi[3] = { frac[0], frac[1], frac[2] };

    int idx[8]; int wgt[8];
    int accumW = 0;
    [unroll] for (int c = 0; c < 8; ++c) {
        int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
        int cx = base[0] + sx; if (cx > nx - 1) cx = nx - 1;
        int cy = base[1] + sy; if (cy > ny - 1) cy = ny - 1;
        int cz = base[2] + sz; if (cz > nz - 1) cz = nz - 1;
        idx[c] = cx + cy * nx + cz * (nx * ny);            // ProbeFlatIndex (cx-major)
        int wx = sx ? whi[0] : wlo[0];
        int wy = sy ? whi[1] : wlo[1];
        int wz = sz ? whi[2] : wlo[2];
        if (c < 7) {
            int64_t p = (int64_t)wx * (int64_t)wy * (int64_t)wz;   // Q48.48
            int wc = (int)(p >> (2 * HF_GI_FRAC));                  // FLOOR narrow -> Q16.16
            wgt[c] = wc;
            accumW += wc;
        } else {
            wgt[c] = HF_GI_ONE - accumW;                  // exact leftover -> Σ w == kOne
        }
    }

    // ===== FxInterpolateSH — int64 8-corner blend, narrow ONCE =====
    // out.coeff[k] = (Σ_c weight_c * cornerSH_c[k]) >> kFrac (Q32.32 accumulator, narrow once).
    int blended[27];
    if (probeCount <= 0) {
        [unroll] for (int z0 = 0; z0 < 27; ++z0) blended[z0] = 0;
    } else {
        [unroll] for (int k = 0; k < 27; ++k) {
            int64_t acc = 0;
            [unroll] for (int c2 = 0; c2 < 8; ++c2)
                acc += (int64_t)wgt[c2] * (int64_t)gSH[idx[c2]].c[k];   // Q32.32 — WIDE
            blended[k] = (int)(acc >> HF_GI_FRAC);                       // narrow ONCE -> Q16.16
        }
    }

    // ===== FxSHEvaluate — the integer cosine-lobe irradiance reconstruction =====
    // SHBasis9 at the normal (the kGiSHBasis generator's formula, live fxmul); IDENTICAL to gi.h::FxSHEvaluate.
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
        int64_t acc = 0;   // Q48.48-ish (coeff*basis*B, three Q16.16 -> Q48.48)
        [unroll] for (int j = 0; j < 9; ++j)
            acc += (int64_t)blended[j * 3 + ch] * (int64_t)Y[j] * B[j];   // WIDE triple product
        int v = (int)(acc >> (2 * HF_GI_FRAC));   // narrow ONCE: Q48.48 >> 32 -> Q16.16
        outc[ch] = (v < 0) ? 0 : v;               // clamp >= 0
    }

    Vec4i res;
    res.x = outc[0]; res.y = outc[1]; res.z = outc[2]; res.w = 0;
    gOut[i] = res;
}
