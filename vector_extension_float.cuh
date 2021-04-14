#ifndef VECTOR_EXTENSION_FLOAT_H
#define VECTOR_EXTENSION_FLOAT_H

#include "vector_extension_utils.cuh"


/////////////////////////////////////
// make_vector
/////////////////////////////////////


inline __host__ __device__ float2 make_float2(float x){
    return make_float2(x, x);
}
inline __host__ __device__ float2 make_float2(float2 v){
    return make_float2(v.x, v.y);
}
inline __host__ __device__ float2 make_float2(float3 v){
    return make_float2(v.x, v.y);
}
inline __host__ __device__ float2 make_float2(float4 v){
    return make_float2(v.x, v.y);
}

inline __host__ __device__ float3 make_float3(float x){
    return make_float3(x, x, x);
}
inline __host__ __device__ float3 make_float3(float x, float y){
    return make_float3(x, y, 0.f);
}
inline __host__ __device__ float3 make_float3(float2 v){
    return make_float3(v.x, v.y, 0.f);
}
inline __host__ __device__ float3 make_float3(float2 v, float x){
    return make_float3(v.x, v.y, x);
}
inline __host__ __device__ float3 make_float3(float x, float2 v){
    return make_float3(x, v.x, v.y);
}
inline __host__ __device__ float3 make_float3(float3 v){
    return make_float3(v.x, v.y, v.z);
}
inline __host__ __device__ float3 make_float3(float4 v){
    return make_float3(v.x, v.y, v.z);
}

inline __host__ __device__ float4 make_float4(float x){
    return make_float4(x, x, x, x);
}
inline __host__ __device__ float4 make_float4(float x, float y){
    return make_float4(x, y, 0.f, 0.f);
}
inline __host__ __device__ float4 make_float4(float2 v){
    return make_float4(v.x, v.y, 0.f, 0.f);
}
inline __host__ __device__ float4 make_float4(float x, float y, float z){
    return make_float4(x, y, z, 0.f);
}
inline __host__ __device__ float4 make_float4(float2 x, float y){
    return make_float4(x.x, x.y, y, 0.f);
}
inline __host__ __device__ float4 make_float4(float x, float2 y){
    return make_float4(x, y.x, y.y, 0.f);
}
inline __host__ __device__ float4 make_float4(float3 v){
    return make_float4(v.x, v.y, v.z, 0.f);
}
inline __host__ __device__ float4 make_float4(float x, float y, float2 z){
    return make_float4(x, y, z.x, z.y);
}
inline __host__ __device__ float4 make_float4(float x, float2 y, float z){
    return make_float4(x, y.x, y.y, z);
}
inline __host__ __device__ float4 make_float4(float2 x, float y, float z){
    return make_float4(x.x, x.y, y, z);
}
inline __host__ __device__ float4 make_float4(float2 x, float2 y){
    return make_float4(x.x, x.y, y.x, y.y);
}
inline __host__ __device__ float4 make_float4(float3 v, float w){
    return make_float4(v.x, v.y, v.z, w);
}
inline __host__ __device__ float4 make_float4(float x, float3 v){
    return make_float4(x, v.x, v.y, v.z);
}
inline __host__ __device__ float4 make_float4(float4 v){
    return make_float4(v.x, v.y, v.z, v.w);
}


/////////////////////////////////////
// negate
/////////////////////////////////////


inline __host__ __device__ float2 operator-(float2 v){ //const float2& v
    return make_float2(-v.x, -v.y);
}

inline __host__ __device__ float3 operator-(float3 v){
    return make_float3(-v.x, -v.y, -v.z);
}

inline __host__ __device__ float4 operator-(float4 v){
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}


/////////////////////////////////////
// +
/////////////////////////////////////


inline __host__ __device__ float2 operator+(float x, float2 y){
    return make_float2(x + y.x, x + y.y);
}
inline __host__ __device__ float2 operator+(float2 x, float y){
    return make_float2(x.x + y, x.y + y);
}
inline __host__ __device__ float2 operator+(float2 x, float2 y){
    return make_float2(x.x + y.x, x.y + y.y);
}
inline __host__ __device__ float3 operator+(float3 x, float y){
    return make_float3(x.x + y, x.y + y, x.z + y);
}
inline __host__ __device__ float3 operator+(float x, float3 y){
    return make_float3(x + y.x, x + y.y, x + y.z);
}
inline __host__ __device__ float3 operator+(float3 x, float2 y){
    return make_float3(x.x + y.x, x.y + y.y, x.z);
}
inline __host__ __device__ float3 operator+(float2 x, float3 y){
    return make_float3(x.x + y.x, x.y + y.y, y.z);
}
inline __host__ __device__ float3 operator+(float3 x, float3 y){
    return make_float3(x.x + y.x, x.y + y.y, x.z + y.z);
}
inline __host__ __device__ float4 operator+(float4 x, float y){
    return make_float4(x.x + y, x.y + y, x.z + y, x.w + y);
}
inline __host__ __device__ float4 operator+(float x, float4 y){
    return make_float4(x + y.x, x + y.y, x + y.z, x + y.w);
}
inline __host__ __device__ float4 operator+(float4 x, float2 y){
    return make_float4(x.x + y.x, x.y + y.y, x.z, x.w);
}
inline __host__ __device__ float4 operator+(float2 x, float4 y){
    return make_float4(x.x + y.x, x.y + y.y, y.z, y.w);
}
inline __host__ __device__ float4 operator+(float4 x, float3 y){
    return make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w);
}
inline __host__ __device__ float4 operator+(float3 x, float4 y){
    return make_float4(x.x + y.x, x.y + y.y, x.z + y.z, y.w);
}
inline __host__ __device__ float4 operator+(float4 x, float4 y){
    return make_float4(x.x + y.x, x.y + y.y, x.z + y.z);
}

// inline __host__ __device__ void operator+=(float2 &x, float x){}
// inline __host__ __device__ void operator+=(float2 &y, float2 y){
//     x.x += y.x;
//     x.y += y.y;
// }

/////////////////////////////////////
// -
/////////////////////////////////////


inline __host__ __device__ float2 operator-(float x, float2 y){
    return make_float2(x - y.x, x - y.y);
}
inline __host__ __device__ float2 operator-(float2 x, float y){
    return make_float2(x.x - y, x.y - y);
}
inline __host__ __device__ float2 operator-(float2 x, float2 y){
    return make_float2(x.x - y.x, x.y - y.y);
}
inline __host__ __device__ float3 operator-(float3 x, float y){
    return make_float3(x.x - y, x.y - y, x.z - y);
}
inline __host__ __device__ float3 operator-(float x, float3 y){
    return make_float3(x - y.x, x - y.y, x - y.z);
}
inline __host__ __device__ float3 operator-(float3 x, float2 y){
    return make_float3(x.x - y.x, x.y - y.y, x.z);
}
inline __host__ __device__ float3 operator-(float2 x, float3 y){
    return make_float3(x.x - y.x, x.y - y.y, y.z);
}
inline __host__ __device__ float3 operator-(float3 x, float3 y){
    return make_float3(x.x - y.x, x.y - y.y, x.z - y.z);
}
inline __host__ __device__ float4 operator-(float4 x, float y){
    return make_float4(x.x - y, x.y - y, x.z - y, x.w - y);
}
inline __host__ __device__ float4 operator-(float x, float4 y){
    return make_float4(x - y.x, x - y.y, x - y.z, x - y.w);
}
inline __host__ __device__ float4 operator-(float4 x, float2 y){
    return make_float4(x.x - y.x, x.y - y.y, x.z, x.w);
}
inline __host__ __device__ float4 operator-(float2 x, float4 y){
    return make_float4(x.x - y.x, x.y - y.y, y.z, y.w);
}
inline __host__ __device__ float4 operator-(float4 x, float3 y){
    return make_float4(x.x - y.x, x.y - y.y, x.z - y.z, x.w);
}
inline __host__ __device__ float4 operator-(float3 x, float4 y){
    return make_float4(x.x - y.x, x.y - y.y, x.z - y.z, y.w);
}
inline __host__ __device__ float4 operator-(float4 x, float4 y){
    return make_float4(x.x - y.x, x.y - y.y, x.z - y.z);
}

// inline __host__ __device__ void operator-=(float2 &x, float2 y)
// {
//     x.x -= y.x;
//     x.y -= y.y;
// }


/////////////////////////////////////
// *
/////////////////////////////////////


inline __host__ __device__ float2 operator*(float x, float2 y){
    return make_float2(x*y.x, x*y.y);
}
inline __host__ __device__ float2 operator*(float2 x, float y){
    return make_float2(x.x*y, x.y*y);
}
inline __host__ __device__ float2 operator*(float2 x, float2 y){
    return make_float2(x.x*y.x, x.y*y.y);
}
inline __host__ __device__ float3 operator*(float3 x, float y){
    return make_float3(x.x*y, x.y*y, x.z*y);
}
inline __host__ __device__ float3 operator*(float x, float3 y){
    return make_float3(x*y.x, x*y.y, x*y.z);
}
inline __host__ __device__ float3 operator*(float3 x, float2 y){
    return make_float3(x.x*y.x, x.y*y.y, x.z);
}
inline __host__ __device__ float3 operator*(float2 x, float3 y){
    return make_float3(x.x*y.x, x.y*y.y, y.z);
}
inline __host__ __device__ float3 operator*(float3 x, float3 y){
    return make_float3(x.x*y.x, x.y*y.y, x.z*y.z);
}
inline __host__ __device__ float4 operator*(float4 x, float y){
    return make_float4(x.x*y, x.y*y, x.z*y, x.w*y);
}
inline __host__ __device__ float4 operator*(float x, float4 y){
    return make_float4(x*y.x, x*y.y, x*y.z, x*y.w);
}
inline __host__ __device__ float4 operator*(float4 x, float2 y){
    return make_float4(x.x*y.x, x.y*y.y, x.z, x.w);
}
inline __host__ __device__ float4 operator*(float2 x, float4 y){
    return make_float4(x.x*y.x, x.y*y.y, y.z, y.w);
}
inline __host__ __device__ float4 operator*(float4 x, float3 y){
    return make_float4(x.x*y.x, x.y*y.y, x.z*y.z, x.w);
}
inline __host__ __device__ float4 operator*(float3 x, float4 y){
    return make_float4(x.x*y.x, x.y*y.y, x.z*y.z, y.w);
}
inline __host__ __device__ float4 operator*(float4 x, float4 y){
    return make_float4(x.x*y.x, x.y*y.y, x.z*y.z);
}


// inline __host__ __device__ void operator*=(float2 &x, float2 y)
// {
//     x.x *= y.x;
//     x.y *= y.y;
// }


/////////////////////////////////////
// /
/////////////////////////////////////


inline __host__ __device__ float2 operator/(float x, float2 y){
    return make_float2(x/y.x, x/y.y);
}
inline __host__ __device__ float2 operator/(float2 x, float y){
    return make_float2(x.x/y, x.y/y);
}
inline __host__ __device__ float2 operator/(float2 x, float2 y){
    return make_float2(x.x/y.x, x.y/y.y);
}
inline __host__ __device__ float3 operator/(float3 x, float y){
    return make_float3(x.x/y, x.y/y, x.z/y);
}
inline __host__ __device__ float3 operator/(float x, float3 y){
    return make_float3(x/y.x, x/y.y, x/y.z);
}
inline __host__ __device__ float3 operator/(float3 x, float2 y){
    return make_float3(x.x/y.x, x.y/y.y, x.z);
}
inline __host__ __device__ float3 operator/(float2 x, float3 y){
    return make_float3(x.x/y.x, x.y/y.y, y.z);
}
inline __host__ __device__ float3 operator/(float3 x, float3 y){
    return make_float3(x.x/y.x, x.y/y.y, x.z/y.z);
}
inline __host__ __device__ float4 operator/(float4 x, float y){
    return make_float4(x.x/y, x.y/y, x.z/y, x.w/y);
}
inline __host__ __device__ float4 operator/(float x, float4 y){
    return make_float4(x/y.x, x/y.y, x/y.z, x/y.w);
}
inline __host__ __device__ float4 operator/(float4 x, float2 y){
    return make_float4(x.x/y.x, x.y/y.y, x.z, x.w);
}
inline __host__ __device__ float4 operator/(float2 x, float4 y){
    return make_float4(x.x/y.x, x.y/y.y, y.z, y.w);
}
inline __host__ __device__ float4 operator/(float4 x, float3 y){
    return make_float4(x.x/y.x, x.y/y.y, x.z/y.z, x.w);
}
inline __host__ __device__ float4 operator/(float3 x, float4 y){
    return make_float4(x.x/y.x, x.y/y.y, x.z/y.z, y.w);
}
inline __host__ __device__ float4 operator/(float4 x, float4 y){
    return make_float4(x.x/y.x, x.y/y.y, x.z/y.z);
}


/////////////////////////////////////
// min
/////////////////////////////////////


inline  __host__ __device__ float2 fminf(float x, float2 y){
    return make_float2(fminf(x,y.x), fminf(x,y.y));
}
inline  __host__ __device__ float2 fminf(float2 x, float y){
    return make_float2(fminf(x.x,y), fminf(x.y,y));
}
inline  __host__ __device__ float2 fminf(float2 x, float2 y){
    return make_float2(fminf(x.x,y.x), fminf(x.y,y.y));
}
inline __host__ __device__ float3 fminf(float x, float3 y){
    return make_float3(fminf(x,y.x), fminf(x,y.y), fminf(x,y.z));
}
inline __host__ __device__ float3 fminf(float3 x, float y){
    return make_float3(fminf(x.x,y), fminf(x.y,y), fminf(x.z,y));
}
inline __host__ __device__ float3 fminf(float3 x, float3 y){
    return make_float3(fminf(x.x,y.x), fminf(x.y,y.y), fminf(x.z,y.z));
}
inline  __host__ __device__ float4 fminf(float x, float4 y){
    return make_float4(fminf(x,y.x), fminf(x,y.y), fminf(x,y.z), fminf(x,y.w));
}
inline  __host__ __device__ float4 fminf(float4 x, float y){
    return make_float4(fminf(x.x,y), fminf(x.y,y), fminf(x.z,y), fminf(x.w,y));
}
inline  __host__ __device__ float4 fminf(float4 x, float4 y){
    return make_float4(fminf(x.x,y.x), fminf(x.y,y.y), fminf(x.z,y.z), fminf(x.w,y.w));
}


/////////////////////////////////////
// max
/////////////////////////////////////


inline  __host__ __device__ float2 fmaxf(float x, float2 y){
    return make_float2(fmaxf(x,y.x), fmaxf(x,y.y));
}
inline  __host__ __device__ float2 fmaxf(float2 x, float y){
    return make_float2(fmaxf(x.x,y), fmaxf(x.y,y));
}
inline  __host__ __device__ float2 fmaxf(float2 x, float2 y){
    return make_float2(fmaxf(x.x,y.x), fmaxf(x.y,y.y));
}
inline __host__ __device__ float3 fmaxf(float x, float3 y){
    return make_float3(fmaxf(x,y.x), fmaxf(x,y.y), fmaxf(x,y.z));
}
inline __host__ __device__ float3 fmaxf(float3 x, float y){
    return make_float3(fmaxf(x.x,y), fmaxf(x.y,y), fmaxf(x.z,y));
}
inline __host__ __device__ float3 fmaxf(float3 x, float3 y){
    return make_float3(fmaxf(x.x,y.x), fmaxf(x.y,y.y), fmaxf(x.z,y.z));
}
inline  __host__ __device__ float4 fmaxf(float x, float4 y){
    return make_float4(fmaxf(x,y.x), fmaxf(x,y.y), fmaxf(x,y.z), fmaxf(x,y.w));
}
inline  __host__ __device__ float4 fmaxf(float4 x, float y){
    return make_float4(fmaxf(x.x,y), fmaxf(x.y,y), fmaxf(x.z,y), fmaxf(x.w,y));
}
inline  __host__ __device__ float4 fmaxf(float4 x, float4 y){
    return make_float4(fmaxf(x.x,y.x), fmaxf(x.y,y.y), fmaxf(x.z,y.z), fmaxf(x.w,y.w));
}


/////////////////////////////////////
// dot
/////////////////////////////////////


inline __host__ __device__ float dot(float2 x, float2 y){
    return x.x*y.x + y.y*y.y;
}
inline __host__ __device__ float dot(float3 x, float3 y){
    return x.x*y.x + x.y*y.y + x.z*y.z;
}
inline __host__ __device__ float dot(float4 x, float4 y){
    return x.x*y.x + x.y*y.y + x.z*y.z + x.w*y.w;
}


/////////////////////////////////////
// cross product
/////////////////////////////////////


inline __host__ __device__ float3 cross(float3 x, float3 y){
    return make_float3(x.y*y.z - x.z*y.y, x.z*y.x - x.x*y.z, x.x*y.y - x.y*y.x);
}


/////////////////////////////////////
// ceil
/////////////////////////////////////


inline __device__ float2 ceilf(float2 v){
    return make_float2(ceilf(v.x), ceilf(v.y));
}
inline __device__ float3 ceilf(float3 v){
    return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}
inline __device__ float4 ceilf(float4 v){
    return make_float4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w));
}


/////////////////////////////////////
// floor
/////////////////////////////////////


inline __host__ __device__ float2 floorf(float2 v){
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v){
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v){
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}


/////////////////////////////////////
// abs
/////////////////////////////////////


inline __host__ __device__ float2 fabs(float2 v){
    return make_float2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ float3 fabs(float3 v){
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ float4 fabs(float4 v){
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}


/////////////////////////////////////
// sum
/////////////////////////////////////


inline __host__ __device__ float sum(float2 x){
    return x.x + x.y;
}
inline __host__ __device__ float sum(float3 x){
    return x.x + x.y + x.z;
}
inline __host__ __device__ float sum(float4 x){
    return x.x + x.y + x.z + x.w;
}


/////////////////////////////////////
// l2-norm
/////////////////////////////////////


inline __host__ __device__ float length(float2 v){
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v){
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v){
    return sqrtf(dot(v, v));
}


/////////////////////////////////////
// clamp
/////////////////////////////////////


inline __device__ __host__ float clamp(float f, float lo, float hi){
    return fmaxf(lo, fminf(f, hi));
}
inline __device__ __host__ float2 clamp(float2 v, float lo, float hi){
    return make_float2(clamp(v.x, lo, hi), clamp(v.y, lo, hi));
}
inline __device__ __host__ float2 clamp(float2 v, float2 lo, float2 hi){
    return make_float2(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y));
}
inline __device__ __host__ float3 clamp(float3 v, float lo, float hi){
    return make_float3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}
inline __device__ __host__ float3 clamp(float3 v, float3 lo, float3 hi){
    return make_float3(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z));
}
inline __device__ __host__ float4 clamp(float4 v, float lo, float hi){
    return make_float4(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi), clamp(v.w, lo, hi));
}
inline __device__ __host__ float4 clamp(float4 v, float4 lo, float4 hi){
    return make_float4(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z), clamp(v.w, lo.w, hi.w));
}


/////////////////////////////////////
// lerp (linear interpolation)
/////////////////////////////////////


inline __device__ __host__ float lerp(float start, float end, float t)
{
    return start + t*(end - start);
}
inline __device__ __host__ float2 lerp(float2 start, float2 end, float t)
{
    return start + t*(end - start);
}
inline __device__ __host__ float3 lerp(float3 start, float3 end, float t)
{
    return start + t*(end - start);
}
inline __device__ __host__ float4 lerp(float4 start, float4 end, float t)
{
    return start + t*(end - start);
}


/////////////////////////////////////
// smooth_step
/////////////////////////////////////


inline __device__ __host__ float smooth_step(float start, float end, float t)
{
    float p = clamp((t - start)/(end - start), 0.f, 1.f);
    return (3.f - 2.f*p)*p*p;
}
inline __device__ __host__ float2 smooth_step(float2 start, float2 end, float2 t)
{
    float2 p = clamp((t - start)/(end - start), 0.f, 1.f);
    return (3.f - 2.f*p)*p*p;
}
inline __device__ __host__ float3 smooth_step(float3 start, float3 end, float3 t)
{
    float3 p = clamp((t - start)/(end - start), 0.f, 1.f);
    return (3.f - 2.f*p)*p*p;
}
inline __device__ __host__ float4 smooth_step(float4 start, float4 end, float4 t)
{
    float4 p = clamp((t - start)/(end - start), 0.f, 1.f);
    return (3.f - 2.f*p)*p*p;
}

#endif