#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H

#ifndef __CUDACC__
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
using namespace std;
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulonglong;

template<typename T>
struct Vec2{
    // static double2 make_vec2(double lo, double hi) { return make_double2(lo, hi); };
};

template<>
struct Vec2<float>{
    typedef float2 vec2;
    static float2 make_vec2(float x, float y){ return make_float2(x, y); };
};

template<>
struct Vec2<double>{
    typedef double2 vec2;
    static double2 make_vec2(double x, double y){ return make_double2(x, y); };
};

template<>
struct Vec2<ulonglong>{
    typedef ulonglong2 vec2;
};



template<typename T>
struct Vec3{
};

template<>
struct Vec3<float>{
    typedef float3 vec3;
    static float3 make_vec3(float x, float y, float z){ return make_float3(x, y, z); };
    static float3 make_vec3(float v){ return make_float3(v, v, v); };
    static float3 make_vec3(double v){ return make_float3(v, v, v); };
    static float3 make_vec3(float3 v){ return v; };
    static float3 make_vec3(double3 v){ return make_float3(v.x, v.y, v.z); };
    static float3 make_vec3(int3 v){ return make_float3((float)v.x, (float)v.y, (float)v.z); };
    static float3 make_vec3(uint3 v){ return make_float3((float)v.x, (float)v.y, (float)v.z); };
};

template<>
struct Vec3<double>{
    typedef double3 vec3;
    static double3 make_vec3(double x, double y, double z){ return make_double3(x, y, z); };
    static double3 make_vec3(float v){ return make_double3(v, v, v); };
    static double3 make_vec3(double v){ return make_double3(v, v, v); };
    static double3 make_vec3(float3 v){ return make_double3(v.x, v.y, v.z); };
    static double3 make_vec3(double3 v){ return v; };
    static double3 make_vec3(int3 v){ return make_double3((double)v.x, (double)v.y, (double)v.z); };
    static double3 make_vec3(uint3 v){ return make_double3((double)v.x, (double)v.y, (double)v.z); };
};

template<>
struct Vec3<ulonglong>{
    typedef ulonglong3 vec3;
};





template<typename T>
struct Vec4{
};

template<>
struct Vec4<float>{
    typedef float4 vec4;
    static float4 make_vec4(float x, float y, float z, float w){ return make_float4(x, y, z, w); };
};

template<>
struct Vec4<double>{
    typedef double4 vec4;
    static double4 make_vec4(double x, double y, double z, double w){ return make_double4(x, y, z, w); };
};


// template<typename T>
// struct vec3{
// };

// template<>
// struct vec3<float>{
// 	typedef float3 vec3_type;
// };

// template<>
// struct vec3<double>{
// 	typedef double3 vec3_type;
// };








// #define _CONCAT(a, b) a ## b
// #define CONCAT(a, b) _CONCAT(a, b)

// #define _VEC2(type) CONCAT(type, 2)
// #define VEC2(type) _VEC2(type)

// #define _VEC3(type) CONCAT(type, 3)
// #define VEC3(type) _VEC3(type)

// #define _VEC4(type) CONCAT(type, 4)
// #define VEC4(type) _VEC4(type)

// #define _MAKEVEC2(type) CONCAT(make_, VEC2(type))
// #define MAKEVEC2(type) _MAKEVEC2(type)

// #define _MAKEVEC3(type) CONCAT(make_, VEC3(type))
// #define MAKEVEC3(type) _MAKEVEC3(type)

// #define _MAKEVEC4(type) CONCAT(make_, VEC4(type))
// #define MAKEVEC4(type) _MAKEVEC4(type)

#endif
