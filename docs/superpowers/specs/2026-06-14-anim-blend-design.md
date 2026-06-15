# Slice X — Animation Blending (design spec)

Date: 2026-06-14
Branch: `slice-anim-blend`

## Goal
Add per-joint cross-clip animation blending to `engine/anim/` (pure C++, hf_core). Two
sampled clips are blended at a scalar weight into a single joint-matrix palette that feeds
the existing skinned render pipeline. No RHI / backend / shader changes.

## Design decisions

### 1. Refactor sampling into a local-pose step
- New POD `struct JointPose { math::Vec3 t; math::Quat r; math::Vec3 s; }` — one local TRS per joint.
- `SampleLocalPose(const Skeleton&, const Animation&, float time) -> std::vector<JointPose>`:
  reproduces the per-channel sampling already in `SampleAnimation` (rest TRS as the base; each
  channel overrides its targeted component; `FindKey` bracket + lerp/slerp/step). Joints with no
  channel keep their rest TRS from the skeleton. **Identical math/order to the old inline loop.**
- `PaletteFromLocalPose(const Skeleton&, const std::vector<JointPose>&) -> std::vector<Mat4>`:
  the existing hierarchy walk (`global[j] = parent? global[parent]*FromTRS : FromTRS`) then
  `palette[j] = global[j] * inverseBind[j]`.
- `SampleAnimation` is re-implemented as `PaletteFromLocalPose(sk, SampleLocalPose(sk, anim, t))`.
  Behaviour is byte-identical (same clamp, same FindKey, same FromTRS, same walk) → the Slice-O
  `skinning` golden MUST NOT change.

### 2. Blend
- `BlendLocalPoses(const std::vector<JointPose>& a, const std::vector<JointPose>& b, float weight)`:
  per joint `t = lerp(a.t,b.t,w)`, `s = lerp(a.s,b.s,w)`, `r = Slerp(a.r,b.r,w)` (Slerp is the
  existing nlerp-with-shortest-arc-flip, already normalized). `weight` clamped to [0,1]. If the two
  pose vectors differ in length, the shorter count is used (defensive; in practice both come from
  the same skeleton).
- `BlendAnimations(skeleton, animA, timeA, animB, timeB, weight)` = `SampleLocalPose` both →
  `BlendLocalPoses` → `PaletteFromLocalPose`. weight 0 → pure A, weight 1 → pure B.

### 3. Showcase + verification
- Vulkan: `hello_triangle.exe --blend-shot <path>` — same scene/camera/light/pipelines as
  `--skinning-shot`, but the palette is a **50/50 blend of "Walk" (t=0.3s) and "Run" (t=0.2s)**.
  (Walk+Run chosen over Survey+Walk: Survey is a near-static sitting/looking pose that visually
  dominates a 50/50 blend; Walk and Run are both locomotion gaits whose 50/50 midpoint is a clearly
  intermediate, legible four-legged stance distinct from either pure clip.) Lit + shadowed, BMP.
- Metal: same showcase added to `metal_headless/visual_test.mm` as `--blend`; new golden
  `tests/golden/metal/anim_blend.png` if the Mac is reachable.

### 4. Unit test (`tests/anim_test.cpp`)
- weight 0 → A's palette exactly; weight 1 → B's palette exactly.
- weight 0.5 of a single-joint translation channel → hand-checked midpoint.
- Slerp of two rotations at 0.5 is normalized (unit length).

## Constraints honoured
- Pure C++ in hf_core; no `vk*`/`MTL`/`Metal` symbols added to the seam (grep stays 12).
- No existing golden re-baked; one NEW golden only.
- `SampleAnimation` refactor byte-identical → `skinning` golden DIFF 0.0000.
