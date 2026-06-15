#include "asset/env_loader.h"

// stb_image is implemented in exactly one TU (engine/asset/gltf_loader.cpp, with
// STB_IMAGE_IMPLEMENTATION). Here we only need the DECLARATIONS of stbi_loadf_from_memory /
// stbi_image_free, so include the header WITHOUT the implementation macro. The implementation TU
// sets STBI_NO_STDIO (decode-from-memory only), so we read the file ourselves and decode from a
// buffer rather than via stbi_loadf(path).
#include "stb/stb_image.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace hf::asset {

namespace {

// IEEE-754 float32 -> float16 (half). Round-to-nearest-even is overkill for an environment map, so
// this is the simpler truncating conversion with correct handling of overflow→inf, subnormals→0,
// and sign/NaN. Good enough for HDR sky radiance (RGBA16F has ~3 decimal digits of mantissa).
uint16_t FloatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;  // rebias 127 -> 15
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFF) == 0xFF) {
        // Inf / NaN: preserve (NaN keeps a non-zero mantissa).
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {
        // Overflow to half: clamp to +/-inf.
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {
        // Subnormal/underflow in half: flush small magnitudes to signed zero. (Env radiance values
        // this tiny are visually black; avoiding subnormal encoding keeps this branch simple.)
        if (exp < -10) return (uint16_t)sign;
        // Encode as a half subnormal.
        mant |= 0x800000u;  // restore implicit 1
        int shift = 14 - exp;
        uint32_t halfMant = mant >> shift;
        return (uint16_t)(sign | halfMant);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

} // namespace

EnvironmentMap LoadHdrEnvironment(rhi::IRHIDevice& device, const char* path) {
    // 1. Read the whole .hdr file into memory (the stb implementation TU sets STBI_NO_STDIO).
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("LoadHdrEnvironment: cannot open ") + path);
    std::streamsize fileSize = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> bytes((size_t)fileSize);
    if (!f.read(reinterpret_cast<char*>(bytes.data()), fileSize))
        throw std::runtime_error(std::string("LoadHdrEnvironment: read failed for ") + path);

    // 2. Decode to linear float RGB (HDR range > 1). Force 3 channels; we add alpha ourselves.
    int w = 0, h = 0, comp = 0;
    float* hdr = stbi_loadf_from_memory(bytes.data(), (int)bytes.size(), &w, &h, &comp, 3);
    if (!hdr)
        throw std::runtime_error(std::string("LoadHdrEnvironment: stbi_loadf failed for ") + path);

    // 3. Build the CPU mip chain by repeated 2x2 box-downsample in float RGB. mip 0 = full-res.
    const int mipLevels = [&] {
        int m = 1, d = (w > h ? w : h);
        while (d > 1) { d >>= 1; ++m; }
        return m;
    }();

    // Per-mip float RGB buffers (mip 0 is the decoded image, copied so we own it after free).
    std::vector<std::vector<float>> mipsF(mipLevels);
    mipsF[0].assign(hdr, hdr + (size_t)w * h * 3);
    stbi_image_free(hdr);

    int pw = w, ph = h;
    for (int i = 1; i < mipLevels; ++i) {
        int cw = pw >> 1; if (cw < 1) cw = 1;
        int ch = ph >> 1; if (ch < 1) ch = 1;
        const std::vector<float>& src = mipsF[i - 1];
        std::vector<float>& dst = mipsF[i];
        dst.resize((size_t)cw * ch * 3);
        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                // Average the (up to) 2x2 parent block. Clamp parent indices for odd dimensions.
                int x0 = x * 2, x1 = (x * 2 + 1 < pw) ? x * 2 + 1 : x * 2;
                int y0 = y * 2, y1 = (y * 2 + 1 < ph) ? y * 2 + 1 : y * 2;
                for (int c = 0; c < 3; ++c) {
                    float a = src[((size_t)y0 * pw + x0) * 3 + c];
                    float b = src[((size_t)y0 * pw + x1) * 3 + c];
                    float d = src[((size_t)y1 * pw + x0) * 3 + c];
                    float e = src[((size_t)y1 * pw + x1) * 3 + c];
                    dst[((size_t)y * cw + x) * 3 + c] = (a + b + d + e) * 0.25f;
                }
            }
        }
        pw = cw; ph = ch;
    }

    // 4. Convert each mip float RGB -> RGBA16F (A = 1.0 half = 0x3C00) and collect pointers.
    std::vector<std::vector<uint16_t>> mipsHalf(mipLevels);
    std::vector<const void*> mipPtrs(mipLevels);
    {
        int mw = w, mh = h;
        for (int i = 0; i < mipLevels; ++i) {
            int cw = mw, ch = mh;
            if (i > 0) { cw = w >> i; if (cw < 1) cw = 1; ch = h >> i; if (ch < 1) ch = 1; }
            const std::vector<float>& srcF = mipsF[i];
            std::vector<uint16_t>& dstH = mipsHalf[i];
            dstH.resize((size_t)cw * ch * 4);
            for (size_t p = 0; p < (size_t)cw * ch; ++p) {
                dstH[p * 4 + 0] = FloatToHalf(srcF[p * 3 + 0]);
                dstH[p * 4 + 1] = FloatToHalf(srcF[p * 3 + 1]);
                dstH[p * 4 + 2] = FloatToHalf(srcF[p * 3 + 2]);
                dstH[p * 4 + 3] = 0x3C00u;  // half 1.0
            }
            mipPtrs[i] = dstH.data();
        }
    }

    // 5. Upload as an N-mip RGBA16F environment texture (desc.environment -> set-3 + env sampler).
    rhi::TextureDesc desc;
    desc.width = (uint32_t)w;
    desc.height = (uint32_t)h;
    desc.format = rhi::Format::RGBA16_Float;
    desc.mipLevels = (uint32_t)mipLevels;
    desc.mipData = mipPtrs.data();
    desc.environment = true;
    // data/dataSize stay null/0; the N-mip path uses mipData.

    EnvironmentMap env;
    env.equirect = device.CreateTexture(desc);
    env.mipLevels = mipLevels;
    env.width = w;
    env.height = h;
    return env;
}

} // namespace hf::asset
