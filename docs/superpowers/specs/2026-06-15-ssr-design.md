# Slice AH — Screen-Space Reflections (SSR) design

Opens Phase 2 (GI/reflections). Classic single-bounce SSR: real reflections of on-screen geometry by
ray-marching the depth buffer in VIEW space and resampling the HDR scene color. ADDITIVE — existing
lit/SSAO/bloom paths, shaders, and the 19 Metal goldens stay byte-for-byte untouched. One new Vulkan
shot (`--ssr-shot`), one new Metal golden (`tests/golden/metal/ssr.png`).

## Reuse (SSR borrows almost everything from SSAO/bloom)

- **G-buffer:** the SAME `gbuffer.{vert,frag}.hlsl` + `gbuffer_instanced.vert.hlsl` (Slice Y) that write
  VIEW-SPACE normal.xyz + LINEAR view depth (`w = -vpos.z`) into an RGBA16F target. SSR reconstructs
  the view-space position from `w` and reads the view-space normal exactly like SSAO does.
- **View<->screen helpers:** SSAO's `ReconstructViewPos(uv, linDepth)` and `ProjectToUV(viewPos)` with
  the `HF_YS` Y-flip sign convention are copied verbatim into `ssr.frag.hlsl`. They are mutual inverses
  and already match the g-buffer rasterization on BOTH backends (the hard part is already solved).
- **HDR scene RT + fullscreen ping-pong:** `CreateRenderTarget(w,h,RGBA16_Float)`, `post.vert`
  fullscreen triangle, `fragmentPushConstants`, `BindTexturePair` (binding 0/1 + 3/4) — identical to
  the bloom/SSAO chains.
- **Tonemap/composite:** the bloom_composite/post.frag ACES + gamma + grade + grain + vignette math is
  replicated verbatim in `ssr_composite.frag.hlsl` (no shared include in the toolchain).

## Passes (all reuse existing RHI — no new RHI types)

1. **shadow** → 2048 directional shadow map (existing static + instanced caster pipelines).
2. **scene → HDR RGBA16F** (`rt`): sky + lit floor (lit.vert/lit.frag) + lit objects. This is the
   radiance SSR samples on a hit.
3. **gbuffer → RGBA16F** (`gbuf`): view-space normal + linear depth (existing gbuffer pipelines).
4. **ssr → RGBA16F** (`ssrRT`): fullscreen. `ssr.frag.hlsl` ray-marches and writes the reflected
   radiance in rgb and the blended reflection weight in `a` (Fresnel * reflectivity * edge/march fade).
   Binds `BindTexturePair(sceneColor, gbuffer)` → scene at t0/s0, gbuffer at t3/s3.
5. **composite → swapchain**: fullscreen. `ssr_composite.frag.hlsl` does
   `final = lerp(scene, ssr.rgb, ssr.a)` then the usual exposure/ACES/grade/vignette.
   Binds `BindTexturePair(sceneColor, ssrRT)`.

## SSR ray-march (the core — `ssr.frag.hlsl`)

For each pixel with a surface (`linDepth > 0`):

1. Reconstruct view-space position `P = ReconstructViewPos(uv, linDepth)` and read view-space normal
   `N = normalize(gbuf.xyz)`.
2. View-space reflection ray: `V = normalize(P)` (camera at origin looks down -Z, so `P` is the
   direction camera→fragment), `R = reflect(V, N)` (HLSL `reflect(I,N) = I - 2*dot(I,N)*N`, I incident).
3. **Reflectivity gate:** reflectivity is derived per-pixel from the surface orientation. The FLOOR is
   the reflective surface; it is (close to) horizontal, so its view-space normal, transformed back, has
   a large world-up component. Rather than pipe a per-pixel material through the g-buffer (which would
   touch gbuffer.frag and its golden), we encode reflectivity from the **fragment's reflection geometry
   + a world-up test reconstructed in view space**: we pass the camera's world-up expressed in VIEW
   space (`viewUp`) via the params, and treat a surface as reflective when `dot(N, viewUp)` is high
   (floor) — objects (spheres/cubes, varied normals) mostly fail this and get ~0 reflectivity. A
   `reflMin/reflMax` smoothstep on `dot(N, viewUp)` yields the reflectivity mask. Documented deviation
   from "pipe a material flag": chosen to keep gbuffer.frag + its golden untouched. The result is the
   same — a mirror floor, non-reflective objects.
4. **March:** fixed `kSteps` (32) linear steps in VIEW space along `R`, total view-space length
   `maxDist` (8.0). At each step project the marched view-space point to UV via `ProjectToUV`, read the
   g-buffer depth there (`sceneDepth`), and compare against the ray's own view depth (`rayDepth`). A HIT
   is when the ray goes from in-front-of to behind the stored surface (`rayDepth > sceneDepth`) within a
   `thickness` (0.5 view units) band — i.e. the surface is between the previous and current sample.
5. **Binary-search refinement:** on the first hit, 5 bisection iterations between the last-miss and
   hit points tighten the intersection UV for a clean (non-banded) reflection.
6. **Fades / fallback:**
   - **Edge fade:** `smoothstep` the hit UV toward the screen borders (fade band 0.12) so reflections
     don't pop at the screen edge where the marched ray exits.
   - **No hit / off-screen / past maxDist:** weight 0 → the composite shows the plain scene (which
     already includes the lit floor's existing procedural IBL sky sheen). Documented: fallback is the
     existing in-scene IBL, NOT a separate cubemap.
   - **Backface/ray-toward-camera:** if `R.z > 0` (ray points back toward the camera, behind the near
     plane) weight 0.
   - **Fresnel:** Schlick `F0=0.04` on `dot(N, -V)` modulates the reflection so grazing angles reflect
     more — multiplied into the final weight along with reflectivity.
7. **Determinism:** no RNG. A fixed 4x4 Bayer-style dither (baked constant) jitters the *start* offset
   by a fraction of one step to hide banding; identical every run → byte-stable golden.

### March params (documented, as shipped)
- `kSteps = 48` linear steps
- `maxDist = 8.0` view-space units
- `thickness = 0.35` view-space units (depth-compare band; tighter = less vertical streak under objects)
- `kRefine = 6` binary-search iterations
- `edgeFade = 0.12` UV border fade band
- `reflMin/reflMax = 0.75 / 0.92` smoothstep on `dot(N, viewUp)` → floor mask
- `kBaseRefl = 0.55` polished-floor base reflectance (used in place of the dielectric 0.04 so the
  reflection reads at near-normal foreground angles); Schlick boosts grazing toward a full mirror.
- The showcase floor uses a DARK low-contrast checker (greys ~18/32) so the mirror reflections read
  clearly instead of washing out against the bright default checkerboard.

### Debug visualization (guarded, no effect on the golden)
`ssr.frag` outputs R=`dot(N,viewUp)`, G=reflectivity, B=hit-weight when `reflMax < 0`; `ssr_composite`
passes the SSR RT through untonemapped when its `pad < 0`. The showcase enables this only under the
`HF_SSR_DBG` env var (off in the golden run), so the captured image is unaffected.

## Params layout (fits existing infra)

`SsrParams` (fragment push constant, `fragmentPushConstants = true`):
```
float2 texel;        // 1/size
float  tanHalfFovY;  // view-space reconstruction
float  aspect;
float  maxDist;
float  thickness;
float  reflMin;
float  reflMax;
float4 viewUp;       // camera world-up expressed in VIEW space (xyz), w unused
```
= 48 bytes, well under push-constant limits. The composite uses a tiny `SsrCompParams { float2 texel;
float intensity; float pad; }` (exposure 1.7).

## Metal V-flip

The whole screen-space march is expressed through `ProjectToUV`/`ReconstructViewPos`, which already
carry the `HF_YS` sign (`-1` Vulkan, `+1` Metal) that makes them mutual inverses AND agree with how the
g-buffer was rasterized under each backend's NDC/texture-origin convention. The marched UVs therefore
sample the right texel on Metal without any extra flip — the `post.vert` `HF_MSL_GEN` V-flip handles
the fullscreen-triangle UV, and `ProjectToUV` handles the per-step projection, exactly as SSAO does.
`viewUp` is a view-space vector (no projection), so it needs no flip. The Metal showcase wraps the
projection in `FlipProjY` like every other Metal showcase, and the g-buffer + SSR pass consume the same
`viewM`, keeping everything self-consistent.

## Showcase scene (`--ssr-shot`)

A flat reflective checkerboard FLOOR (large scaled plane at y=0) with several DISTINCT colored objects
sitting ON it — colored cubes + spheres at known x/z, plus the DamagedHelmet if available — lit by the
existing directional light + shadow. Fixed camera looking down at a grazing angle so the floor fills the
lower screen and the objects' inverted reflections are clearly visible below them. Deterministic.

## Unit test (`tests/ssr_test.cpp`, `engine/render/ssr.h`)

`engine/render/ssr.h` (header-only, hf_core, no backend) factors the two pure helpers the shader uses:
`ViewToScreenUV(viewPos, tanHalfFovY, aspect, yFlip)` and `ReflectView(incident, normal)`. The test
asserts: (a) a known view-space point projects to the expected screen UV and round-trips through
`ReconstructViewPos`; (b) `ReflectView` of a known incidence about a known normal equals the analytic
reflection (e.g. straight-down ray off a horizontal floor reflects straight up). ASan-eligible.

## Verification

- Vulkan: `hello_triangle --ssr-shot ssr.bmp` → BMP→PNG → visually inspect: floor shows recognizable
  inverted mirror reflections of the objects, correctly positioned under each object, edge-faded, no
  black floor / detached / smeared reflections.
- Metal: `visual_test --ssr tests/golden/metal/ssr.png`, two-run DIFF 0.0000; the 19 existing goldens
  stay DIFF 0.0000. `scripts/verify.ps1` golden list 19→20. (If the Mac is unreachable, Metal code
  lands and the golden is "pending".)
