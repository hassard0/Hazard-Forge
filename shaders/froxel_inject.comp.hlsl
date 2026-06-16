// Slice CS — Froxel Volumetric Fog: the INJECT compute pass. ONE thread per froxel (dimX*dimY*dimZ),
// dispatched as enough HF_FROXEL_THREADS workgroups to cover the volume. Each thread reconstructs its
// froxel's VIEW-space center (the screen-tile NDC center unprojected via invProj at the slice's center
// view-distance — the EXACT mirror of render::froxel::SliceCenterViewZ + cluster.h's UnprojectToView),
// transforms it to WORLD space (via invView) to evaluate the height-fog Density, computes the sun in-
// scatter (sunColor * Phase(cos(viewDir,sunDir), g) * density) and the extinction (density), and writes
// (scatterRGB, extinction) into float4 #0 of the FLAT SSBO froxel volume cell [fx + fy*dimX + fz*dimX*dimY].
//
// Slice CX — Volumetric Shadows (sun light shafts through the fog): if volumetricShadows (gParams.range.z)
// is set, AFTER the sun in-scatter this pass samples the sun's CSM shadow map at the froxel's WORLD center
// (froxel::SunVisibility mirrored VERBATIM): project the world center by the cascade's lightViewProj
// (cascade chosen by the froxel's camera-forward view depth, mirroring lit_csm.frag's split pick), read
// the stored occluder depth from the shadow map (the SAME sampled depth texture the lit pass uses, bound
// to this compute pass via the additive BindShadowMapCompute seam), and compare with a fixed bias. The
// resulting visibility (1 lit / 0 shadowed) MULTIPLIES the SUN in-scatter ONLY (the CV clustered-light
// in-scatter is UNAFFECTED) -> a froxel in the sun's shadow gets no sun scatter (dark fog volume), the lit
// froxels between occluders read as bright foggy sun shafts. With volumetricShadows=false this whole gate
// is SKIPPED (sun visibility forced to 1) -> the code is the EXACT CV path -> the byte-identical
// shadows-off == CV proof. It is the SUN scatter (already ×density) that is gated, so density=0 -> sun
// scatter 0 regardless -> the density=0 == no-fog proof still holds.
//
// Slice CV — Per-Froxel Clustered-Light Injection (the CS+CL fusion): if injectLights (gParams.clusterDims.w)
// is set, AFTER the sun in-scatter this pass maps the froxel to its CLUSTER (XY tile from the froxel XY,
// which align — both 16x9; Z via SliceForViewZ of the froxel center's view-Z, cluster dimZ=24), reads
// clusterLightGrid[clusterIdx] = {count, idx[96]}, iterates that cluster's lights from the SAME light SSBO
// the clustered-lights (CL) pass uses (VIEW-space pos), and ADDS each light's windowed-attenuation * HG
// phase * density in-scatter (render::froxel::InjectClusteredLights mirrored VERBATIM). This is a PURELY
// ADDITIVE term: with injectLights=false the loop is SKIPPED and the code is the EXACT CS path -> the
// lights-off render is BYTE-IDENTICAL to the CS sun-only froxel fog (the lights-off proof). It is also
// ×density, so at baseDensity=0 the whole scatter is 0 -> the density=0 == no-fog proof still holds.
//
// The math (exponential Z slices, NDC-corner unprojection via invProj, height density, HG phase, the
// windowed point-light attenuation) is the verbatim mirror of engine/render/froxel.h + cluster.h. With
// baseDensity==0 every cell's density is 0 -> scatter 0, extinction 0 -> the integrate pass is a per-step
// no-op -> the apply is a pure pass-through (the zero-density == no-fog-scene proof).
//
// Buffers (storage, bound at compute bindings 0..3; on Metal they land at buffer(kCsStorage+0..3)):
//   b0 gVolume          : the flat FroxelCell[dimX*dimY*dimZ] volume, WRITE (float4 #0 = scatter+ext).
//   b1 gParams          : grid dims + zNear/zFar + sun + fog + cluster dims/range + injectLights, READ.
//   b2 gClusterLightGrid: the per-cluster {count, idx[96]} ordered light lists the CL pass fills, READ.
//   b3 gLights          : the point lights (VIEW-space pos+radius, color+intensity) the CL pass uses, READ.
// (b2/b3 are bound for both the lights-off and lights-on renders; the injectLights=false path never reads
//  them — the showcase binds the dummy buffer there for the CS-equivalent render — so the disabled path
//  is the EXACT CS code.)
//   t4/s4 gSunShadow/gSunShadowSmp (Slice CX): the sun's CSM shadow depth texture + its sampler, bound to
//     this COMPUTE pass via the additive RHI BindShadowMapCompute seam (the SAME sampled depth map the lit
//     pass samples). Only READ when volumetricShadows is set; the showcase always binds a valid shadow map
//     so the disabled (volumetricShadows=false) path — which never samples it — is the EXACT CV code.
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL are the HF_MSL_GEN
// generation-path guards + [[vk::binding]] decorations (same as cluster_assign / gtao.frag), not backend
// CODE symbols. spirv-cross maps the SPIR-V bindings to Metal's flat compute buffer indices so the SAME
// HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross): bit-identical math.

#define HF_FROXEL_THREADS 64
#define HF_MAX_LIGHTS_PER_CLUSTER 96

static const float HF_PI = 3.14159265358979323846;

// One froxel cell (std430, 32 bytes): mirrors render::froxel::FroxelCell. float4 #0 = injected scatter +
// extinction (this pass writes it); float4 #1 = integrated in-scatter + transmittance (integrate writes).
struct FroxelCell {
    float4 scatterExt;   // xyz = in-scatter radiance, w = extinction
    float4 resultT;      // xyz = integrated in-scatter, w = transmittance
};

// Froxel-grid + camera + fog params (std430). Shared by all three froxel passes (the integrate pass only
// reads dims+range; the trailing CV cluster fields are appended after the CS layout, so the integrate
// pass — which never reads them — is unaffected and froxel_fog stays byte-identical).
struct FroxelParams {
    uint4    dims;          // x=dimX, y=dimY, z=dimZ, w=unused
    float4   range;         // x=zNear, y=zFar, z/w unused
    float4   sunDir;        // xyz = directional-light TRAVEL direction (world space), w unused
    float4   sunColor;      // rgb = sun color * intensity, w unused
    float4   fog;           // x=baseDensity, y=heightFalloff, z=heightRef, w=g (HG anisotropy)
    float4x4 invProj;       // proj.Inverse(), column-major (recovers view-space XY from NDC)
    float4x4 invView;       // view.Inverse(), column-major (view -> world for density height)
    // --- Slice CV cluster-light-injection params (appended; CS-only renders leave injectLights 0). ---
    uint4    clusterDims;   // x=cDimX, y=cDimY, z=cDimZ, w=injectLights (1=add clustered lights, 0=CS)
    float4   clusterRange;  // x=cZNear, y=cZFar, z/w unused (the cluster grid's exponential Z slicing)
    float4x4 view;          // world -> view, column-major (transforms cluster math + the apply's view ray)
    // --- Slice CX volumetric-shadow params (appended; volumetricShadows=false leaves them unread). ---
    uint4    shadowFlags;   // x=volumetricShadows (1=gate sun by CSM shadow, 0=CV path), y=numCascades, zw unused
    float4   csmSplits;     // x,y,z,w = view-space FAR distance of cascade 0..3 (lit_csm.frag's splits)
    float4   shadowBias;    // x=depth bias (anti-acne, mirrors lit.frag bias=0.0025), yzw unused
    float4   camFwd;        // xyz = camera FORWARD (world), for the cascade view-depth pick; w unused
    float4   camPos;        // xyz = camera/eye world position (view-depth = dot(wpos-camPos, camFwd)); w unused
    float4x4 cascadeVP[4];  // per-cascade sun lightViewProj (Ortho*LookAt); cascade 0 == the single map
};

// One cluster's fixed-cap ordered light list — matches cluster_assign.comp / lit_clustered_cl.frag
// (std430): uint count + uint pad[3] + uint idx[96].
struct ClusterList {
    uint count;
    uint pad0, pad1, pad2;
    uint idx[HF_MAX_LIGHTS_PER_CLUSTER];
};

// One point light — matches the CL light SSBO (GpuLight): VIEW-space pos + radius, color + intensity.
struct GpuLight {
    float4 posRadius;   // xyz = VIEW-space position, w = radius
    float4 color;       // rgb = color, w = intensity
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FroxelCell>   gVolume          : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FroxelParams> gParams          : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<ClusterList>  gClusterLightGrid : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<GpuLight>     gLights          : register(u3);
// Slice CX: the sun's CSM shadow depth texture + sampler (the SAME map the lit pass samples), bound to
// this COMPUTE pass at compute-set bindings 4/5 via the additive BindShadowMapCompute seam. Only read
// when volumetricShadows is set. spirv-cross maps these to the Metal compute texture(0)/sampler(0) slots.
[[vk::binding(4, 0)]] Texture2D    gSunShadow    : register(t4);
[[vk::binding(5, 0)]] SamplerState gSunShadowSmp : register(s4);

// SliceZ(k): positive view-distance of exponential slice boundary at REAL k. Mirrors froxel.h::SliceZ.
float SliceZ(float k, uint dimZ, float zNear, float zFar) {
    return zNear * pow(zFar / zNear, k / (float)dimZ);
}

// Unproject NDC (nx,ny) at positive view-distance vz to view space, via invProj. Mirrors
// cluster.h/froxel.h UnprojectToView: form the NDC ray (z=0), scale to the plane z = -vz.
float3 UnprojectToView(float4x4 invProj, float nx, float ny, float vz) {
    float4 h = mul(invProj, float4(nx, ny, 0.0, 1.0));
    float3 vDir = h.xyz / h.w;
    float  s = (abs(vDir.z) > 1e-9) ? (-vz / vDir.z) : 0.0;
    return float3(vDir.x * s, vDir.y * s, -vz);
}

// Height-fog density at a world point. Mirrors froxel.h::Density EXACTLY (baseDensity==0 -> 0 EXACTLY).
float Density(float3 worldPos, float baseDensity, float heightFalloff, float heightRef) {
    if (baseDensity == 0.0) return 0.0;
    float d = baseDensity * exp(-heightFalloff * (worldPos.y - heightRef));
    return d > 0.0 ? d : 0.0;
}

// Henyey-Greenstein phase. Mirrors froxel.h::Phase (forward-peaked g>0, isotropic g=0, finite).
float Phase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    if (denom < 1e-6) denom = 1e-6;
    return (1.0 - g2) / (4.0 * HF_PI * denom * sqrt(denom));
}

// Positive view-distance -> cluster z-slice index. Mirrors cluster.h::SliceForViewZ EXACTLY (the froxel
// center's view-Z maps to its cluster's exponential Z slice; XY tiles already align). Used by the CV
// light loop to find each froxel's cluster.
uint ClusterSliceForViewZ(float viewZ, uint cDimZ, float cZNear, float cZFar) {
    if (viewZ <= cZNear) return 0u;
    if (viewZ >= cZFar)  return cDimZ - 1u;
    float t = log(viewZ / cZNear) / log(cZFar / cZNear);   // [0,1)
    int s = (int)floor(t * (float)cDimZ);
    return (uint)(s < 0 ? 0 : (s > (int)cDimZ - 1 ? (int)cDimZ - 1 : s));
}

// Slice CX cascade selection — mirrors render::froxel::SelectCascade + lit_csm.frag: smallest cascade
// whose split far-distance still contains the view depth; last cascade is the catch-all. numCascades==1
// always returns 0 (the single sun shadow map == one cascade, the showcase case).
uint SelectCascade(float viewDepth, float4 splits, uint numCascades) {
    uint cascade = numCascades - 1u;
    [unroll] for (uint ci = 0u; ci < 4u; ++ci) {
        if (ci < numCascades && viewDepth <= splits[ci]) { cascade = ci; return cascade; }
    }
    return cascade;
}

// Slice CX sun visibility — mirrors render::froxel::SunVisibility VERBATIM (and the lit pass's shadow
// sample): project worldPos by the cascade lightViewProj, derive smUV/curDepth, sample the stored
// occluder depth, compare with the bias. Returns 1 (LIT) / 0 (SHADOWED); out-of-cascade -> 1.
float SunVisibility(float3 worldPos, float4x4 sunViewProj, float bias) {
    float4 lp = mul(sunViewProj, float4(worldPos, 1.0));
    float3 proj = lp.xyz / lp.w;
    float2 smUV = proj.xy * 0.5 + 0.5;
    // Metal-only shadow-map texture-origin flip (same as lit.frag / lit_csm.frag): lightViewProj has the
    // Vulkan Y-flip baked in; on Metal it's flipped CPU-side so V must be flipped to hit the matching texel.
#ifdef HF_MSL_GEN
    smUV.y = 1.0 - smUV.y;
#endif
    float curDepth = proj.z;
    // DEGENERATE / out-of-cascade -> LIT (1): off the shadow map or behind the light's near/far plane.
    if (smUV.x < 0.0 || smUV.x > 1.0 || smUV.y < 0.0 || smUV.y > 1.0 || curDepth < 0.0 || curDepth > 1.0)
        return 1.0;
    // SampleLevel (explicit LOD 0), not Sample: a compute shader has no implicit derivatives, so an
    // implicit-LOD sample is invalid here. The shadow map is single-mip, so LOD 0 == the lit pass's sample.
    float occluder = gSunShadow.SampleLevel(gSunShadowSmp, smUV, 0.0).r;
    return (curDepth - bias > occluder) ? 0.0 : 1.0;
}

[numthreads(HF_FROXEL_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint dimX = gParams[0].dims.x;
    uint dimY = gParams[0].dims.y;
    uint dimZ = gParams[0].dims.z;
    float zNear = gParams[0].range.x;
    float zFar  = gParams[0].range.y;
    float4x4 invProj = gParams[0].invProj;
    float4x4 invView = gParams[0].invView;
    float3 sunDirWorld = normalize(gParams[0].sunDir.xyz);   // light TRAVEL direction (world)
    float3 sunColor = gParams[0].sunColor.rgb;
    float baseDensity = gParams[0].fog.x;
    float heightFalloff = gParams[0].fog.y;
    float heightRef = gParams[0].fog.z;
    float g = gParams[0].fog.w;

    uint froxelCount = dimX * dimY * dimZ;
    uint c = gid.x;
    if (c >= froxelCount) return;

    // Decode (fx,fy,fz) from the flat index: idx = fx + fy*dimX + fz*(dimX*dimY).
    uint fz  = c / (dimX * dimY);
    uint rem = c % (dimX * dimY);
    uint fy  = rem / dimX;
    uint fx  = rem % dimX;

    // The froxel's view-space CENTER: the screen-tile NDC center unprojected at the slice center view-z.
    float nx = (((float)fx + 0.5) / (float)dimX) * 2.0 - 1.0;
    float ny = (((float)fy + 0.5) / (float)dimY) * 2.0 - 1.0;
    float vzCenter = SliceZ((float)fz + 0.5, dimZ, zNear, zFar);   // SliceCenterViewZ
    float3 viewCenter = UnprojectToView(invProj, nx, ny, vzCenter);

    // View -> world for the height-based density (invView is affine; w stays 1).
    float3 worldCenter = mul(invView, float4(viewCenter, 1.0)).xyz;
    float density = Density(worldCenter, baseDensity, heightFalloff, heightRef);

    // Sun single-scatter in-scatter: the cosine between the VIEW ray (froxel center -> eye, i.e. toward
    // the camera = -normalize(viewCenter)) and the direction TOWARD the sun (-sunDir). Looking toward the
    // sun through the fog -> cosTheta near 1 -> a forward-peaked phase glow.
    float3 toCamera = normalize(-viewCenter);
    // Transform the world sun travel-dir into view space via invView's inverse is the view matrix; but we
    // only need the COSINE, which is rotation-invariant — so compute it in WORLD space: the world view ray
    // is worldCenter -> camera. Reconstruct the world ray as the world-space direction of -viewCenter.
    float3 viewRayWorld = normalize(mul(invView, float4(toCamera, 0.0)).xyz);  // world dir toward camera
    float3 toSunWorld = normalize(-sunDirWorld);                               // world dir toward the sun
    float cosTheta = dot(viewRayWorld, toSunWorld);
    float phase = Phase(cosTheta, g);

    float3 scatter = sunColor * (phase * density);
    float extinction = density;

    // --- Slice CX: gate the SUN in-scatter by the sun's CSM shadow visibility at this froxel's world
    // center (sun-only; the CV clustered lights below are UNAFFECTED). volumetricShadows=false SKIPS this
    // entirely -> the EXACT CV path (the byte-identical shadows-off == CV proof). ---
    uint volumetricShadows = gParams[0].shadowFlags.x;
    if (volumetricShadows != 0u) {
        uint numCascades = gParams[0].shadowFlags.y;
        float bias = gParams[0].shadowBias.x;
        // Cascade pick by the froxel's camera-forward view depth (mirrors lit_csm.frag).
        float3 camFwd = normalize(gParams[0].camFwd.xyz);
        float3 camPos = gParams[0].camPos.xyz;
        float viewDepth = dot(worldCenter - camPos, camFwd);
        uint cascade = SelectCascade(viewDepth, gParams[0].csmSplits, numCascades);
        float sunVis = SunVisibility(worldCenter, gParams[0].cascadeVP[cascade], bias);
        scatter *= sunVis;   // multiplicative gate on the SUN scatter only
    }

    // --- Slice CV: ADD the clustered point lights' in-scatter (purely additive; skipped when off). ---
    uint injectLights = gParams[0].clusterDims.w;
    if (injectLights != 0u) {
        uint cDimX = gParams[0].clusterDims.x;
        uint cDimY = gParams[0].clusterDims.y;
        uint cDimZ = gParams[0].clusterDims.z;
        float cZNear = gParams[0].clusterRange.x;
        float cZFar  = gParams[0].clusterRange.y;

        // Map the froxel to its cluster: XY tiles align (froxel fx/fy == cluster cx/cy, both 16x9); the
        // Z slice is the cluster's exponential slice for the froxel center's view-Z (vzCenter is the
        // positive view-distance). idx = cx + cy*cDimX + cz*(cDimX*cDimY) (cluster cx-major order).
        uint cx = fx;
        uint cy = fy;
        uint cz = ClusterSliceForViewZ(vzCenter, cDimZ, cZNear, cZFar);
        uint clusterIdx = cx + cy * cDimX + cz * (cDimX * cDimY);

        // The view-space direction the camera ray travels for this froxel (eye at origin -> froxel),
        // mirroring render::froxel::InjectClusteredLights's viewDir == normalize(froxelViewPos).
        float3 viewDir = normalize(viewCenter);

        uint count = gClusterLightGrid[clusterIdx].count;
        [loop] for (uint i = 0; i < count; ++i) {
            uint li = gClusterLightGrid[clusterIdx].idx[i];
            float4 pr = gLights[li].posRadius;
            float3 lvpos  = pr.xyz;          // VIEW-space light position (the CL light SSBO)
            float  radius = pr.w;
            float3 lcolor    = gLights[li].color.rgb;
            float  intensity = gLights[li].color.w;

            float3 toLight = lvpos - viewCenter;
            float  dist = length(toLight);

            // WINDOWED HARD-RADIUS attenuation IDENTICAL to CL (lit_clustered_cl.frag): 0 at d>=radius.
            float r4  = dist / radius; r4 = r4 * r4; r4 = r4 * r4;   // (d/radius)^4
            float win = saturate(1.0 - r4); win = win * win;
            float atten = (1.0 / (dist * dist + 0.01)) * win;

            float  invD = (dist > 1e-9) ? (1.0 / dist) : 0.0;
            float  cosL = dot(viewDir, toLight * invD);
            float  phaseL = Phase(cosL, g);

            scatter += lcolor * (intensity * atten * phaseL * density);
        }
    }

    gVolume[c].scatterExt = float4(scatter, extinction);
    // Leave resultT untouched here; the integrate pass overwrites it. (Initialize for safety/determinism.)
    gVolume[c].resultT = float4(0.0, 0.0, 0.0, 1.0);
}
