#pragma once
#include <cmath>
#include <cstdint>

namespace hf::math {

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator*(const Vec3& a, float s) { return {a.x*s, a.y*s, a.z*s}; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
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

} // namespace hf::math
