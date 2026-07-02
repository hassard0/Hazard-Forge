// Physical atmospheric scattering (Slice AT1) — the fullscreen 3-panel sky showcase pass. Each
// panel is a panorama (horizontal = azimuth centered on the sun azimuth, vertical = view elevation
// 0..89 deg) evaluated by the SAME Rayleigh + Henyey-Greenstein-Mie single-scattering integral as
// engine/render/atmosphere.h (the math is mirrored VERBATIM below — same constants, same fixed
// kViewSamples x kSunSamples loop counts — the clouds.frag shared-math discipline). The three sun
// elevations (noon / low / sunset) arrive as push-constant sun directions computed CPU-side from the
// documented atmosphere.h panel constants; the shader tonemaps (1-exp(-c*exposure), gamma 1/2.2 —
// TonemapSky mirrored) and writes the final LDR panel directly. DETERMINISTIC: fixed constants,
// fixed loop counts, pure function of uv + push constants (no clock, no RNG, no textures) -> two
// runs byte-identical; the CPU test pins the same math. EXISTING sky/clouds shaders + goldens are
// untouched (new standalone pass).
struct AtmoParams {
    float3 sunDirA; float exposure;   // panel 0 (noon)
    float3 sunDirB; float pad0;       // panel 1 (low)
    float3 sunDirC; float pad1;       // panel 2 (sunset)
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer AtmoPC { AtmoParams ap; };
#define HF_AP ap
#else
[[vk::push_constant]] struct { AtmoParams p; } pc;
#define HF_AP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float HF_PI = 3.14159265358979323846;

// --- atmosphere.h math, mirrored VERBATIM (must stay identical to engine/render/atmosphere.h). ------
static const float kPlanetRadius     = 6360e3;
static const float kAtmosphereRadius = 6420e3;
static const float kHr = 7994.0;
static const float kHm = 1200.0;
static const float3 kBetaR = float3(5.8e-6, 13.5e-6, 33.1e-6);
static const float kBetaM = 21e-6;
static const float kMieExtinction = 1.1;
static const float kMieG = 0.76;
static const float kSunIntensity = 20.0;
static const float kEyeHeight = 2.0;
static const int kViewSamples = 16;
static const int kSunSamples  = 8;
static const float kPanelAzSpanRad  = 3.490659;   // 200 deg
static const float kPanelElevMaxRad = 1.553343;   //  89 deg

float RayleighPhase(float cosTheta) {
    return 3.0 / (16.0 * HF_PI) * (1.0 + cosTheta * cosTheta);
}

float MiePhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    denom = max(denom, 1e-6);
    return (1.0 - g2) / (4.0 * HF_PI * denom * sqrt(denom));
}

float RaySphereFar(float3 ro, float3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    return -b + sqrt(disc);
}

float RaySphereNear(float3 ro, float3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    return -b - sqrt(disc);
}

// Panorama view direction for panel-local (u01, v01) — matches atmo::PanoramaViewDir.
float3 PanoramaViewDir(float u01, float v01) {
    float az = (u01 - 0.5) * kPanelAzSpanRad;
    float el = v01 * kPanelElevMaxRad;
    float ce = cos(el);
    return float3(sin(az) * ce, sin(el), -cos(az) * ce);
}

// The single-scattering sky — matches atmo::SkyColor (same loop counts, same accumulators).
float3 SkyColor(float3 viewDir, float3 sunDir) {
    float3 ro = float3(0.0, kPlanetRadius + kEyeHeight, 0.0);

    float tMax = RaySphereFar(ro, viewDir, kAtmosphereRadius);
    if (tMax <= 0.0) return float3(0.0, 0.0, 0.0);
    float tGround = RaySphereNear(ro, viewDir, kPlanetRadius);
    if (tGround > 0.0 && tGround < tMax) tMax = tGround;

    float cosTheta = dot(viewDir, sunDir);
    float phR = RayleighPhase(cosTheta);
    float phM = MiePhase(cosTheta, kMieG);

    float seg = tMax / (float)kViewSamples;
    float odR = 0.0, odM = 0.0;
    float3 sumR = float3(0.0, 0.0, 0.0);
    float3 sumM = float3(0.0, 0.0, 0.0);

    [loop] for (int i = 0; i < kViewSamples; ++i) {
        float t = ((float)i + 0.5) * seg;
        float3 p = ro + viewDir * t;
        float h = length(p) - kPlanetRadius;
        h = max(h, 0.0);
        float hr = exp(-h / kHr) * seg;
        float hm = exp(-h / kHm) * seg;
        odR += hr;
        odM += hm;

        float tSun = RaySphereFar(p, sunDir, kAtmosphereRadius);
        if (tSun <= 0.0) continue;
        float segS = tSun / (float)kSunSamples;
        float odRs = 0.0, odMs = 0.0;
        bool shadowed = false;
        [loop] for (int j = 0; j < kSunSamples; ++j) {
            float ts = ((float)j + 0.5) * segS;
            float3 ps = p + sunDir * ts;
            float hs = length(ps) - kPlanetRadius;
            if (hs < 0.0) { shadowed = true; break; }
            odRs += exp(-hs / kHr) * segS;
            odMs += exp(-hs / kHm) * segS;
        }
        if (shadowed) continue;

        float me = kBetaM * kMieExtinction * (odM + odMs);
        float3 att = exp(-(kBetaR * (odR + odRs) + float3(me, me, me)));
        sumR += att * hr;
        sumM += att * hm;
    }

    return (sumR * kBetaR * phR + sumM * kBetaM * phM) * kSunIntensity;
}

// Exposure tonemap + gamma — matches atmo::TonemapSky.
float3 TonemapSky(float3 c, float exposure) {
    float3 m = float3(1.0, 1.0, 1.0) - exp(-c * exposure);
    float g = 1.0 / 2.2;
    return pow(max(m, float3(0.0, 0.0, 0.0)), float3(g, g, g));
}
// --- end mirrored math -------------------------------------------------------------------------------

float4 main(PSInput i) : SV_Target {
    // Three vertical panels across the screen: noon / low / sunset (left to right).
    float x3 = i.uv.x * 3.0;
    int panel = (int)min(x3, 2.999);
    float u01 = x3 - (float)panel;          // panel-local horizontal 0..1
    float v01 = 1.0 - i.uv.y;               // 0 = horizon (bottom), 1 = near-zenith (top)

    // Thin black separators between panels so the triptych reads.
    if (panel > 0 && u01 < 0.008) return float4(0.0, 0.0, 0.0, 1.0);

    float3 sunDir = (panel == 0) ? HF_AP.sunDirA : ((panel == 1) ? HF_AP.sunDirB : HF_AP.sunDirC);
    float3 viewDir = PanoramaViewDir(u01, v01);
    float3 c = SkyColor(viewDir, normalize(sunDir));
    return float4(TonemapSky(c, HF_AP.exposure), 1.0);
}
