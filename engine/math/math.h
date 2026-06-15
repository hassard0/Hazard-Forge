#pragma once
#include <cmath>
#include <cstdint>

namespace hf::math {

struct Vec2 {
    float x = 0, y = 0;
    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
};

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator*(const Vec3& a, float s) { return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }              // unary negate
inline Vec3 operator/(const Vec3& a, float s) { return {a.x/s, a.y/s, a.z/s}; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    return len > 0 ? Vec3{v.x/len, v.y/len, v.z/len} : v;
}

// Column-major 4x4: element(row, col) == m[col*4 + row].
struct Mat4 {
    float m[16] = {0};

    static Mat4 Identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    // Right-handed, depth range [0,1] (Vulkan), Y flipped for Vulkan clip space.
    static Mat4 Perspective(float fovYRad, float aspect, float zNear, float zFar) {
        float t = std::tan(fovYRad * 0.5f);
        Mat4 r;  // all zeros
        r.m[0]  = 1.0f / (aspect * t);       // col0,row0
        r.m[5]  = -1.0f / t;                  // col1,row1 (negative = Vulkan Y flip)
        r.m[10] = zFar / (zNear - zFar);      // col2,row2
        r.m[11] = -1.0f;                      // col2,row3
        r.m[14] = (zNear * zFar) / (zNear - zFar); // col3,row2
        return r;
    }

    // Orthographic projection, depth range [0,1] (Vulkan), Y flipped to match Perspective.
    static Mat4 Ortho(float l, float r, float b, float t, float zn, float zf) {
        Mat4 m;  // zeros; column-major
        m.m[0]  = 2.0f / (r - l);
        m.m[5]  = -2.0f / (t - b);                 // Y-flip (matches Perspective's -1/tan)
        m.m[10] = -1.0f / (zf - zn);
        m.m[12] = -(r + l) / (r - l);
        m.m[13] = (t + b) / (t - b);               // sign paired with the Y-flip
        m.m[14] = -zn / (zf - zn);
        m.m[15] = 1.0f;
        return m;
    }

    static Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        Vec3 f = normalize(center - eye);
        Vec3 s = normalize(cross(f, up));
        Vec3 u = cross(s, f);
        Mat4 r = Identity();
        r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;   // row0 = s
        r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;   // row1 = u
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z; // row2 = -f
        r.m[12] = -dot(s, eye);
        r.m[13] = -dot(u, eye);
        r.m[14] = dot(f, eye);
        return r;
    }

    static Mat4 Translate(const Vec3& t) {
        Mat4 r = Identity();
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }

    static Mat4 RotateY(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 r = Identity();
        r.m[0] = c;  r.m[8] = s;
        r.m[2] = -s; r.m[10] = c;
        return r;
    }

    static Mat4 RotateX(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 r = Identity();
        r.m[5] = c;  r.m[9] = -s;
        r.m[6] = s;  r.m[10] = c;
        return r;
    }

    static Mat4 RotateZ(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 r = Identity();
        r.m[0] = c;  r.m[4] = -s;   // element(0,0)=c, element(0,1)=-s
        r.m[1] = s;  r.m[5] = c;    // element(1,0)=s, element(1,1)=c
        return r;
    }

    static Mat4 Scale(const Vec3& s) {
        Mat4 r = Identity();
        r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
        return r;
    }

    // General 4x4 inverse (adjugate / cofactor method). Defined out-of-line below operator*.
    Mat4 Inverse() const;
};

// C = A * B  (standard matrix product; C(row,col) = sum_k A(row,k)*B(k,col)).
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 c;  // zeros
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float sum = 0;
            for (int k = 0; k < 4; ++k)
                sum += a.m[k*4 + row] * b.m[col*4 + k];
            c.m[col*4 + row] = sum;
        }
    return c;
}

// General 4x4 inverse via the adjugate (cofactor) method, column-major. Used by the editor's
// screen-ray unprojection (inverse view-proj). If the matrix is singular (|det| ~ 0) the identity
// is returned so callers degrade gracefully rather than producing NaNs. Verified by the unit test
// (M * M.Inverse() == I) for identity, translation, and a full TRS matrix.
inline Mat4 Mat4::Inverse() const {
    const float* a = m;
    float inv[16];
    inv[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2]  =  a[1]*a[6]*a[15]  - a[1]*a[7]*a[14]  - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7]  - a[13]*a[3]*a[6];
    inv[6]  = -a[0]*a[6]*a[15]  + a[0]*a[7]*a[14]  + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7]  + a[12]*a[3]*a[6];
    inv[10] =  a[0]*a[5]*a[15]  - a[0]*a[7]*a[13]  - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]  + a[0]*a[6]*a[13]  + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5];
    inv[3]  = -a[1]*a[6]*a[11]  + a[1]*a[7]*a[10]  + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]   + a[9]*a[3]*a[6];
    inv[7]  =  a[0]*a[6]*a[11]  - a[0]*a[7]*a[10]  - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]   - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]  + a[0]*a[7]*a[9]   + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]   + a[8]*a[3]*a[5];
    inv[15] =  a[0]*a[5]*a[10]  - a[0]*a[6]*a[9]   - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]   - a[8]*a[2]*a[5];

    float det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    if (det > -1e-12f && det < 1e-12f) return Mat4::Identity();
    float invDet = 1.0f / det;
    Mat4 r;
    for (int i = 0; i < 16; ++i) r.m[i] = inv[i] * invDet;
    return r;
}

// Transform a point (w=1) by a column-major Mat4: p' = M * [p,1] (no perspective divide). Named
// MulPoint (not TransformPoint) to avoid colliding with the file-local helper in debug_draw.cpp.
inline Vec3 MulPoint(const Mat4& m, const Vec3& p) {
    return {
        m.m[0]*p.x + m.m[4]*p.y + m.m[8] *p.z + m.m[12],
        m.m[1]*p.x + m.m[5]*p.y + m.m[9] *p.z + m.m[13],
        m.m[2]*p.x + m.m[6]*p.y + m.m[10]*p.z + m.m[14],
    };
}

// Transform a point through a clip-space matrix and apply the perspective divide (w). Returns the
// post-divide xyz; `outW` exposes the clip w so callers can detect points behind the camera (w<=0).
inline Vec3 MulPointDivide(const Mat4& m, const Vec3& p, float& outW) {
    float x = m.m[0]*p.x + m.m[4]*p.y + m.m[8] *p.z + m.m[12];
    float y = m.m[1]*p.x + m.m[5]*p.y + m.m[9] *p.z + m.m[13];
    float z = m.m[2]*p.x + m.m[6]*p.y + m.m[10]*p.z + m.m[14];
    float w = m.m[3]*p.x + m.m[7]*p.y + m.m[11]*p.z + m.m[15];
    outW = w;
    float inv = (w > 1e-9f || w < -1e-9f) ? 1.0f / w : 1.0f;
    return {x*inv, y*inv, z*inv};
}

// --- Ray + intersection math (editor picking / gizmo hit-testing). -----------------------------
// All deterministic and window-free; unit-tested in tests/editor_test.cpp.

struct Ray {
    Vec3 origin;
    Vec3 dir;  // assumed normalized (use MakeRay to build from two points)
};

// Ray from `origin` toward `target` (direction normalized).
inline Ray MakeRay(const Vec3& origin, const Vec3& target) {
    return Ray{origin, normalize(target - origin)};
}

// Axis-aligned bounding box.
struct Aabb {
    Vec3 min;
    Vec3 max;
};

// Slab ray/AABB test. On a hit, `tHit` is the nearest non-negative entry parameter along the ray
// (0 if the origin is inside the box). Returns false if the box is entirely behind the origin or
// missed. dir components of 0 are handled via the standard inf-slope slab convention.
inline bool RayAabb(const Ray& r, const Aabb& box, float& tHit) {
    float tmin = -1e30f, tmax = 1e30f;
    const float o[3] = {r.origin.x, r.origin.y, r.origin.z};
    const float d[3] = {r.dir.x, r.dir.y, r.dir.z};
    const float mn[3] = {box.min.x, box.min.y, box.min.z};
    const float mx[3] = {box.max.x, box.max.y, box.max.z};
    for (int i = 0; i < 3; ++i) {
        if (d[i] > -1e-9f && d[i] < 1e-9f) {
            if (o[i] < mn[i] || o[i] > mx[i]) return false;  // parallel & outside this slab
        } else {
            float inv = 1.0f / d[i];
            float t1 = (mn[i] - o[i]) * inv;
            float t2 = (mx[i] - o[i]) * inv;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return false;
        }
    }
    if (tmax < 0.0f) return false;       // box entirely behind the ray origin
    tHit = (tmin > 0.0f) ? tmin : 0.0f;  // inside the box -> 0
    return true;
}

// Ray/sphere intersection (nearest non-negative root). On a hit, `tHit` = that parameter.
inline bool RaySphere(const Ray& r, const Vec3& center, float radius, float& tHit) {
    Vec3 oc = r.origin - center;
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return false;
    float s = std::sqrt(disc);
    float t = -b - s;
    if (t < 0.0f) t = -b + s;
    if (t < 0.0f) return false;
    tHit = t;
    return true;
}

// Closest approach between an (infinite) ray and the segment a->b. Returns the segment parameter
// `segT` in [0,1] of the closest point on the segment; `outRayT` (>=0, clamped) is the ray
// parameter; `outDist` is the world distance between the two closest points. Used for gizmo axis
// hit-testing and translate-drag projection. Based on the standard closest-points-of-two-lines
// solve with the segment clamped to [0,1] and the ray clamped to t>=0.
inline float RayClosestParamToSegment(const Ray& r, const Vec3& a, const Vec3& b,
                                      float& outRayT, float& outDist) {
    Vec3 u = r.dir;            // ray direction (unit)
    Vec3 v = b - a;            // segment direction (not unit)
    Vec3 w0 = r.origin - a;
    float A = dot(u, u);       // == 1
    float B = dot(u, v);
    float C = dot(v, v);
    float D = dot(u, w0);
    float E = dot(v, w0);
    float denom = A * C - B * B;
    float sc, tc;  // sc = ray param, tc = segment param
    if (denom > -1e-9f && denom < 1e-9f) {
        // Ray and segment parallel: clamp the segment param to the ray-origin projection.
        sc = 0.0f;
        tc = (C > 1e-9f) ? (E / C) : 0.0f;
    } else {
        sc = (B * E - C * D) / denom;
        tc = (A * E - B * D) / denom;
    }
    if (sc < 0.0f) sc = 0.0f;          // ray cannot go behind its origin
    if (tc < 0.0f) tc = 0.0f;          // clamp to the segment
    if (tc > 1.0f) tc = 1.0f;
    Vec3 pRay = r.origin + u * sc;
    Vec3 pSeg = a + v * tc;
    outRayT = sc;
    outDist = length(pRay - pSeg);
    return tc;
}

// --- Quaternion (x,y,z,w), used for skeletal-animation rotation channels. ----------------------
// Identity = (0,0,0,1). Stored the same way glTF stores ROTATION channels (xyzw), so the loader
// copies the four floats straight in. Slerp is implemented as normalized-lerp (nlerp): for the
// small per-frame angular deltas of a sampled animation it is visually indistinguishable from true
// slerp, is branch-light, and preserves the shortest-arc behaviour via the dot-sign flip. (See the
// design spec — nlerp is the documented, accepted choice for this slice.)
struct Quat {
    float x = 0, y = 0, z = 0, w = 1;
    Quat() = default;
    Quat(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    static Quat Identity() { return Quat{0, 0, 0, 1}; }
};

inline Quat Normalize(const Quat& q) {
    float len = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (len <= 0.0f) return Quat::Identity();
    float inv = 1.0f / len;
    return Quat{q.x*inv, q.y*inv, q.z*inv, q.w*inv};
}

// Normalized-lerp between two unit quaternions, taking the shortest arc (flip b if dot < 0).
// t is clamped to [0,1] by the caller (the sampler clamps the keyframe fraction).
inline Quat Slerp(const Quat& a, Quat b, float t) {
    float d = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    if (d < 0.0f) { b.x = -b.x; b.y = -b.y; b.z = -b.z; b.w = -b.w; }
    Quat r{a.x + (b.x - a.x) * t,
           a.y + (b.y - a.y) * t,
           a.z + (b.z - a.z) * t,
           a.w + (b.w - a.w) * t};
    return Normalize(r);
}

// Column-major rotation matrix from a (assumed-normalized) quaternion. Matches the RH convention
// used elsewhere: element(row,col) == m[col*4 + row].
inline Mat4 QuatToMat4(const Quat& q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    Mat4 r = Mat4::Identity();
    r.m[0]  = 1.0f - 2.0f*(yy + zz);  // col0,row0
    r.m[1]  = 2.0f*(xy + wz);         // col0,row1
    r.m[2]  = 2.0f*(xz - wy);         // col0,row2
    r.m[4]  = 2.0f*(xy - wz);         // col1,row0
    r.m[5]  = 1.0f - 2.0f*(xx + zz);  // col1,row1
    r.m[6]  = 2.0f*(yz + wx);         // col1,row2
    r.m[8]  = 2.0f*(xz + wy);         // col2,row0
    r.m[9]  = 2.0f*(yz - wx);         // col2,row1
    r.m[10] = 1.0f - 2.0f*(xx + yy);  // col2,row2
    return r;
}

// Compose a local transform from translation, rotation (quaternion) and scale: T * R * S.
// (Scale applied first, then rotate, then translate — the standard glTF node-TRS order.)
inline Mat4 FromTRS(const Vec3& t, const Quat& r, const Vec3& s) {
    Mat4 m = QuatToMat4(r);
    // Fold the scale into the rotation columns (column c is scaled by s[c]).
    m.m[0] *= s.x; m.m[1] *= s.x; m.m[2] *= s.x;
    m.m[4] *= s.y; m.m[5] *= s.y; m.m[6] *= s.y;
    m.m[8] *= s.z; m.m[9] *= s.z; m.m[10] *= s.z;
    // Translation in the last column.
    m.m[12] = t.x; m.m[13] = t.y; m.m[14] = t.z;
    return m;
}

// Advance an orientation quaternion by a world-space angular velocity over dt, using the standard
// first-order rigid-body update q' = normalize(q + 0.5 * (0,omega) * q * dt). The pure-quaternion
// (0, omega) is multiplied on the LEFT (world-frame angular velocity), matching the semi-implicit
// Euler integrator in engine/physics. Renormalizes to keep q unit. Reusable, RNG/clock-free.
inline Quat IntegrateOrientation(const Quat& q, const Vec3& omega, float dt) {
    // Quaternion product (0,omega) * q, with the omega quaternion w-component = 0.
    float wx = omega.x, wy = omega.y, wz = omega.z;
    Quat dq{
        wy*q.z - wz*q.y + wx*q.w,   // x
        wz*q.x - wx*q.z + wy*q.w,   // y
        wx*q.y - wy*q.x + wz*q.w,   // z
        -(wx*q.x + wy*q.y + wz*q.z) // w
    };
    float h = 0.5f * dt;
    Quat r{q.x + dq.x*h, q.y + dq.y*h, q.z + dq.z*h, q.w + dq.w*h};
    return Normalize(r);
}

} // namespace hf::math
