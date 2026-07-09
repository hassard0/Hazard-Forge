// Hazard Forge — DETERMINISTIC procedural noise for the material graph (Slice MG1).
//
// The MOAT twist: this noise is a LIVE INTEGER HASH (no host-baked LUT, no texture/binding) that the
// CPU interpreter in engine/material/shader_graph.cpp replicates BIT-FOR-BIT. The hash uses only uint
// xor / shift / multiply — which wrap mod 2^32 IDENTICALLY in C++ and HLSL/MSL — plus a power-of-two
// float divide (exact in IEEE), so hfHash2f(cell) is bit-exact CPU==GPU and the interpolated noise
// matches to float tolerance (the visresolve bar). No int64, no groupshared, no dynamic array
// indexing, bounded loops only -> spirv-cross emits clean MSL (the "MSL-gen-safe" requirement).
//
// These functions are the TEXTUAL TWIN of Eval{ValueNoise,Perlin,Voronoi,Fbm} in shader_graph.cpp —
// the two MUST stay in lockstep (same constants, same op order). The material codegen emits an
// #include of THIS file only when a noise node is present, so the existing (noise-free) material
// goldens are byte-unchanged.
#ifndef HF_MATERIAL_NOISE_HLSLI
#define HF_MATERIAL_NOISE_HLSLI

// Low-bias integer avalanche (the CPU HfHashU twin).
uint hfHashU(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
// 2D integer cell hash. Cells are biased by +1024 before the uint cast so small negative neighbour
// cells convert identically to the CPU (modular int->uint conversion matches C++).
uint hfHash2i(int ix, int iy) {
    uint ux = (uint)(ix + 1024);
    uint uy = (uint)(iy + 1024);
    return hfHashU(hfHashU(ux * 0x9e3779b1u) ^ (uy * 0x85ebca77u));
}
// Cell hash -> [0,1). Mask to 24 bits then divide by 2^24 (exact power-of-two float divide).
float hfHash2f(int ix, int iy) {
    return float(hfHash2i(ix, iy) & 0xFFFFFFu) * (1.0 / 16777216.0);
}

// Smoothstep value noise -> [0,1].
float hfValueNoise(float2 p) {
    float fpx = floor(p.x), fpy = floor(p.y);
    int ix = (int)fpx, iy = (int)fpy;
    float fx = p.x - fpx, fy = p.y - fpy;
    float ux = fx * fx * (3.0 - 2.0 * fx);
    float uy = fy * fy * (3.0 - 2.0 * fy);
    float a = hfHash2f(ix, iy),     b = hfHash2f(ix + 1, iy);
    float c = hfHash2f(ix, iy + 1), d = hfHash2f(ix + 1, iy + 1);
    float ab = a + (b - a) * ux;
    float cd = c + (d - c) * ux;
    return ab + (cd - ab) * uy;
}

// Gradient/Perlin noise. Gradients are unit vectors from an angle hash (no dynamic array indexing).
float hfPerlinGrad(int gx, int gy, float dx, float dy) {
    float ang = hfHash2f(gx, gy) * 6.28318530718;
    return cos(ang) * dx + sin(ang) * dy;
}
float hfPerlin(float2 p) {
    float fpx = floor(p.x), fpy = floor(p.y);
    int ix = (int)fpx, iy = (int)fpy;
    float fx = p.x - fpx, fy = p.y - fpy;
    float ux = fx * fx * (3.0 - 2.0 * fx);
    float uy = fy * fy * (3.0 - 2.0 * fy);
    float va = hfPerlinGrad(ix,     iy,     fx,        fy);
    float vb = hfPerlinGrad(ix + 1, iy,     fx - 1.0,  fy);
    float vc = hfPerlinGrad(ix,     iy + 1, fx,        fy - 1.0);
    float vd = hfPerlinGrad(ix + 1, iy + 1, fx - 1.0,  fy - 1.0);
    float ab = va + (vb - va) * ux;
    float cd = vc + (vd - vc) * ux;
    float n = ab + (cd - ab) * uy;
    return n * 0.5 + 0.5;
}

// Cellular / Voronoi F1 nearest-feature distance -> [0,~1.4].
float hfVoronoi(float2 p) {
    float fpx = floor(p.x), fpy = floor(p.y);
    int ix = (int)fpx, iy = (int)fpy;
    float fx = p.x - fpx, fy = p.y - fpy;
    float md = 8.0;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            int cx = ix + ox, cy = iy + oy;
            float rx = float(ox) + hfHash2f(cx, cy) - fx;
            float ry = float(oy) + hfHash2f(cy, cx) - fy;   // swapped args = independent y jitter
            float d = rx * rx + ry * ry;
            md = min(md, d);
        }
    }
    return sqrt(md);
}

// Fractal Brownian motion (fractal sum of ValueNoise). Bounded 8-octave loop; `octaves` caps it.
float hfFbm(float2 p, int octaves) {
    float sum = 0.0, amp = 0.5, freq = 1.0, norm = 0.0;
    for (int i = 0; i < 8; ++i) {
        if (i >= octaves) break;
        sum += amp * hfValueNoise(p * freq);
        norm += amp;
        amp *= 0.5; freq *= 2.0;
    }
    return (norm > 1e-6) ? sum / norm : 0.0;
}

#endif // HF_MATERIAL_NOISE_HLSLI
