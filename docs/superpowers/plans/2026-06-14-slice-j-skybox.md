# Slice J: Procedural Skybox — Plan

Builds on A–I. Adds a gradient sky + sun behind the scene. Pure shader+sample+UBO change
(reuses the Slice G fullscreen pipeline path; no new RHI surface).

## Approach
Draw a fullscreen "sky" triangle FIRST in the scene→RT pass (depth test OFF, depth write OFF), then
draw geometry over it. The sky shader reconstructs a world-space view ray from the camera basis
(passed in the UBO — NO matrix inverse) and outputs a horizon→zenith gradient plus a sun glow.

## FrameData additions (keep C++/lit.vert/lit.frag/shadow.vert/sky.* in sync; now 352B < 512 cap)
Append after lightViewProj: `float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;`
(skyParams.x = tanHalfFov, .y = aspect). C++: camFwd[4], camRight[4], camUp[4], skyParams[4].

## Shaders
- `sky.vert.hlsl`: fullscreen triangle from SV_VertexID (same as post.vert), pass uv (0..1).
- `sky.frag.hlsl`: cbuffer Frame at set0 b0 (the FrameData). Reconstruct ray:
  `float2 ndc = uv*2-1; float3 ray = normalize(f.camFwd.xyz + f.camRight.xyz*ndc.x*f.skyParams.x*f.skyParams.y + f.camUp.xyz*(-ndc.y)*f.skyParams.x);`
  Gradient: `float h = saturate(ray.y*0.5+0.5); float3 zenith=float3(0.18,0.30,0.62); float3 horizon=float3(0.65,0.72,0.82); float3 sky=lerp(horizon,zenith,pow(h,0.8));`
  Sun: `float3 sunDir = normalize(-f.lightDir.xyz); float s = pow(max(dot(ray,sunDir),0.0),256.0); sky += float3(1.0,0.95,0.8)*s*2.0;`
  Lower hemisphere (ray.y<0): fade toward a dim ground haze. Output float4(sky,1).
  (Verify the ndc.y sign against the --shot; flip if the gradient is upside down.)

## Pipeline
Sky pipeline: `fullscreen=true, usesFrameUniforms=true, usesTexture=false, depthTest=false`,
colorFormat=swapchain format, empty vertexLayout, pushConstantSize=0.

## Sample
- Compute camera basis each frame: fwd=normalize(center-eye); right=normalize(cross(fwd,{0,1,0}));
  up=cross(right,fwd). Fill camFwd/camRight/camUp; skyParams={tan(0.5*60deg), aspect}.
  (Add Vec3 cross/normalize if not already in math — they exist in math.h.)
- In the scene→RT pass: after BeginRenderPass(clear) and BEFORE drawScene: bind sky pipeline,
  SetFrameUniforms already called, Draw(3). Then drawScene as before. (Sky writes no depth, so
  geometry with depth-test draws over it.)
- Both --shot and interactive paths.

## Verify
Build clean; ctest (rhi_smoke+math_test) pass; `--shot` exit 0, BMP ~3.6MB; the captured image shows
a sky gradient + sun behind the lit/shadowed scene (no longer a dark void). grep: no vk in rhi/ or scene/.

## Commit
`feat: procedural skybox (gradient + sun) via camera-ray reconstruction`
