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

} // namespace hf::math
