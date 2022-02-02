#ifndef VECTOR_EXTENSION_H
#define VECTOR_EXTENSION_H


#include "vector_extension_float.h"
#include "vector_extension_double.h"
#include "vector_extension_int.h"
#include "vector_extension_longlong.h"
#include "vector_extension_uint.h"
#include "vector_extension_ulonglong.h"


/////////////////////////////////////
// make_vector
/////////////////////////////////////


inline __host__ __device__ float2 make_float2(int x){
    return make_float2((float)x, (float)x);
}
inline __host__ __device__ float2 make_float2(int x, int y){
    return make_float2((float)x, (float)y);
}
inline __host__ __device__ float2 make_float2(int2 v){
    return make_float2((float)v.x, (float)v.y);
}
inline __host__ __device__ float2 make_float2(int3 v){
    return make_float2((float)v.x, (float)v.y);
}
inline __host__ __device__ float2 make_float2(int4 v){
    return make_float2((float)v.x, (float)v.y);
}

inline __host__ __device__ float2 make_float2(uint x){
    return make_float2((float)x, (float)x);
}
inline __host__ __device__ float2 make_float2(uint x, uint y){
    return make_float2((float)x, (float)y);
}
inline __host__ __device__ float2 make_float2(uint2 v){
    return make_float2((float)v.x, (float)v.y);
}
inline __host__ __device__ float2 make_float2(uint3 v){
    return make_float2((float)v.x, (float)v.y);
}
inline __host__ __device__ float2 make_float2(uint4 v){
    return make_float2((float)v.x, (float)v.y);
}

inline __host__ __device__ float2 make_float2(double x){
    return make_float2((float)x, (float)x);
}
inline __host__ __device__ float2 make_float2(double x, double y){
    return make_float2((float)x, (float)y);
}
inline __host__ __device__ float2 make_float2(double2 v){
    return make_float2((float)v.x, (float)v.y);
}
inline __host__ __device__ float2 make_float2(double3 v){
    return make_float2((float)v.x, (float)v.y);
}
inline __host__ __device__ float2 make_float2(double4 v){
    return make_float2((float)v.x, (float)v.y);
}

inline __host__ __device__ float3 make_float3(int x){
    return make_float3((float)x, (float)x, (float)x);
}
inline __host__ __device__ float3 make_float3(int x, int y){
    return make_float3((float)x, (float)y, 0.f);
}
inline __host__ __device__ float3 make_float3(int2 v){
    return make_float3((float)v.x, (float)v.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(int2 v, int x){
    return make_float3((float)v.x, (float)v.y, (float)x);
}
inline __host__ __device__ float3 make_float3(int x, int2 v){
    return make_float3((float)x, (float)v.x, (float)v.y);
}
inline __host__ __device__ float3 make_float3(int x, int y, int z){
    return make_float3((float)x, (float)y, (float)z);
}
inline __host__ __device__ float3 make_float3(int3 v){
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}
inline __host__ __device__ float3 make_float3(int4 v){
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}

inline __host__ __device__ float3 make_float3(uint x){
    return make_float3((float)x, (float)x, (float)x);
}
inline __host__ __device__ float3 make_float3(uint x, uint y){
    return make_float3((float)x, (float)y, 0.f);
}
inline __host__ __device__ float3 make_float3(uint2 v){
    return make_float3((float)v.x, (float)v.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(uint2 v, uint x){
    return make_float3((float)v.x, (float)v.y, (float)x);
}
inline __host__ __device__ float3 make_float3(uint x, uint2 v){
    return make_float3((float)x, (float)v.x, (float)v.y);
}
inline __host__ __device__ float3 make_float3(uint x, uint y, uint z){
    return make_float3((float)x, (float)y, (float)z);
}
inline __host__ __device__ float3 make_float3(uint3 v){
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}
inline __host__ __device__ float3 make_float3(uint4 v){
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}

inline __host__ __device__ float3 make_float3(double x){
    return make_float3((float)x, (float)x, (float)x);
}
inline __host__ __device__ float3 make_float3(double x, double y){
    return make_float3((float)x, (float)y, 0.f);
}
inline __host__ __device__ float3 make_float3(double2 v){
    return make_float3((float)v.x, (float)v.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(double2 v, double x){
    return make_float3((float)v.x, (float)v.y, (float)x);
}
inline __host__ __device__ float3 make_float3(double x, double2 v){
    return make_float3((float)x, (float)v.x, (float)v.y);
}
inline __host__ __device__ float3 make_float3(double x, double y, double z){
    return make_float3((float)x, (float)y, (float)z);
}
inline __host__ __device__ float3 make_float3(double3 v){
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}
inline __host__ __device__ float3 make_float3(double4 v){
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}


inline __host__ __device__ float4 make_float4(int x){
    return make_float4((float)x, (float)x, (float)x, (float)x);
}
inline __host__ __device__ float4 make_float4(int x, int y){
    return make_float4((float)x, (float)y, 0.f, 0.f);
}
inline __host__ __device__ float4 make_float4(int2 v){
    return make_float4((float)v.x, (float)v.y, 0.f, 0.f);
}
inline __host__ __device__ float4 make_float4(int x, int y, int z){
    return make_float4((float)x, (float)y, (float)z, 0.f);
}
inline __host__ __device__ float4 make_float4(int2 x, int y){
    return make_float4((float)x.x, (float)x.y, (float)y, 0.f);
}
inline __host__ __device__ float4 make_float4(int x, int2 y){
    return make_float4((float)x, (float)y.x, (float)y.y, 0.f);
}
inline __host__ __device__ float4 make_float4(int3 v){
    return make_float4((float)v.x, (float)v.y, (float)v.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(int x, int y, int z, int w){
    return make_float4((float)x, (float)y, (float)z, (float)w);
}
inline __host__ __device__ float4 make_float4(int x, int y, int2 z){
    return make_float4((float)x, (float)y, (float)z.x, (float)z.y);
}
inline __host__ __device__ float4 make_float4(int x, int2 y, int z){
    return make_float4((float)x, (float)y.x, (float)y.y, (float)z);
}
inline __host__ __device__ float4 make_float4(int2 x, int y, int z){
    return make_float4((float)x.x, (float)x.y, (float)y, (float)z);
}
inline __host__ __device__ float4 make_float4(int2 x, int2 y){
    return make_float4((float)x.x, (float)x.y, (float)y.x, (float)y.y);
}
inline __host__ __device__ float4 make_float4(int3 v, int w){
    return make_float4((float)v.x, (float)v.y, (float)v.z, (float)w);
}
inline __host__ __device__ float4 make_float4(int x, int3 v){
    return make_float4((float)x, (float)v.x, (float)v.y, (float)v.z);
}
inline __host__ __device__ float4 make_float4(int4 v){
    return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

inline __host__ __device__ float4 make_float4(uint x){
    return make_float4((float)x, (float)x, (float)x, (float)x);
}
inline __host__ __device__ float4 make_float4(uint x, uint y){
    return make_float4((float)x, (float)y, 0.f, 0.f);
}
inline __host__ __device__ float4 make_float4(uint2 v){
    return make_float4((float)v.x, (float)v.y, 0.f, 0.f);
}
inline __host__ __device__ float4 make_float4(uint x, uint y, uint z){
    return make_float4((float)x, (float)y, (float)z, 0.f);
}
inline __host__ __device__ float4 make_float4(uint2 x, uint y){
    return make_float4((float)x.x, (float)x.y, (float)y, 0.f);
}
inline __host__ __device__ float4 make_float4(uint x, uint2 y){
    return make_float4((float)x, (float)y.x, (float)y.y, 0.f);
}
inline __host__ __device__ float4 make_float4(uint3 v){
    return make_float4((float)v.x, (float)v.y, (float)v.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(uint x, uint y, uint z, uint w){
    return make_float4((float)x, (float)y, (float)z, (float)w);
}
inline __host__ __device__ float4 make_float4(uint x, uint y, uint2 z){
    return make_float4((float)x, (float)y, (float)z.x, (float)z.y);
}
inline __host__ __device__ float4 make_float4(uint x, uint2 y, uint z){
    return make_float4((float)x, (float)y.x, (float)y.y, (float)z);
}
inline __host__ __device__ float4 make_float4(uint2 x, uint y, uint z){
    return make_float4((float)x.x, (float)x.y, (float)y, (float)z);
}
inline __host__ __device__ float4 make_float4(uint2 x, uint2 y){
    return make_float4((float)x.x, (float)x.y, (float)y.x, (float)y.y);
}
inline __host__ __device__ float4 make_float4(uint3 v, uint w){
    return make_float4((float)v.x, (float)v.y, (float)v.z, (float)w);
}
inline __host__ __device__ float4 make_float4(uint x, uint3 v){
    return make_float4((float)x, (float)v.x, (float)v.y, (float)v.z);
}
inline __host__ __device__ float4 make_float4(uint4 v){
    return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}



inline __host__ __device__ double3 make_double3(int x){
    return make_double3((double)x, (double)x, (double)x);
}
inline __host__ __device__ double3 make_double3(int x, int y){
    return make_double3((double)x, (double)y, 0.0);
}
inline __host__ __device__ double3 make_double3(int2 v){
    return make_double3((double)v.x, (double)v.y, 0.0);
}
inline __host__ __device__ double3 make_double3(int2 v, int x){
    return make_double3((double)v.x, (double)v.y, (double)x);
}
inline __host__ __device__ double3 make_double3(int x, int2 v){
    return make_double3((double)x, (double)v.x, (double)v.y);
}
inline __host__ __device__ double3 make_double3(int x, int y, int z){
    return make_double3((double)x, (double)y, (double)z);
}
inline __host__ __device__ double3 make_double3(int3 v){
    return make_double3((double)v.x, (double)v.y, (double)v.z);
}
inline __host__ __device__ double3 make_double3(int4 v){
    return make_double3((double)v.x, (double)v.y, (double)v.z);
}

inline __host__ __device__ double3 make_double3(uint x){
    return make_double3((double)x, (double)x, (double)x);
}
inline __host__ __device__ double3 make_double3(uint x, uint y){
    return make_double3((double)x, (double)y, 0.0);
}
inline __host__ __device__ double3 make_double3(uint2 v){
    return make_double3((double)v.x, (double)v.y, 0.0);
}
inline __host__ __device__ double3 make_double3(uint2 v, uint x){
    return make_double3((double)v.x, (double)v.y, (double)x);
}
inline __host__ __device__ double3 make_double3(uint x, uint2 v){
    return make_double3((double)x, (double)v.x, (double)v.y);
}
inline __host__ __device__ double3 make_double3(uint x, uint y, uint z){
    return make_double3((double)x, (double)y, (double)z);
}
inline __host__ __device__ double3 make_double3(uint3 v){
    return make_double3((double)v.x, (double)v.y, (double)v.z);
}
inline __host__ __device__ double3 make_double3(uint4 v){
    return make_double3((double)v.x, (double)v.y, (double)v.z);
}





inline __host__ __device__ int2 make_int2(uint x){
    return make_int2((int)x, (int)x);
}
inline __host__ __device__ int2 make_int2(uint x, uint y){
    return make_int2((int)x, (int)y);
}
inline __host__ __device__ int2 make_int2(uint2 v){
    return make_int2((int)v.x, (int)v.y);
}
inline __host__ __device__ int2 make_int2(uint3 v){
    return make_int2((int)v.x, (int)v.y);
}
inline __host__ __device__ int2 make_int2(uint4 v){
    return make_int2((int)v.x, (int)v.y);
}

inline __host__ __device__ int2 make_int2(float x){
    return make_int2((int)x, (int)x);
}
inline __host__ __device__ int2 make_int2(float x, float y){
    return make_int2((int)x, (int)y);
}
inline __host__ __device__ int2 make_int2(float2 v){
    return make_int2((int)v.x, (int)v.y);
}
inline __host__ __device__ int2 make_int2(float3 v){
    return make_int2((int)v.x, (int)v.y);
}
inline __host__ __device__ int2 make_int2(float4 v){
    return make_int2((int)v.x, (int)v.y);
}

inline __host__ __device__ int3 make_int3(uint x){
    return make_int3((int)x, (int)x, (int)x);
}
inline __host__ __device__ int3 make_int3(uint x, uint y){
    return make_int3((int)x, (int)y, 0);
}
inline __host__ __device__ int3 make_int3(uint2 v){
    return make_int3((int)v.x, (int)v.y, 0);
}
inline __host__ __device__ int3 make_int3(uint2 v, uint x){
    return make_int3((int)v.x, (int)v.y, (int)x);
}
inline __host__ __device__ int3 make_int3(uint x, uint2 v){
    return make_int3((int)x, (int)v.x, (int)v.y);
}
inline __host__ __device__ int3 make_int3(uint x, uint y, uint z){
    return make_int3((int)x, (int)y, (int)z);
}
inline __host__ __device__ int3 make_int3(uint3 v){
    return make_int3((int)v.x, (int)v.y, (int)v.z);
}
inline __host__ __device__ int3 make_int3(uint4 v){
    return make_int3((int)v.x, (int)v.y, (int)v.z);
}

inline __host__ __device__ int3 make_int3(float x){
    return make_int3((int)x, (int)x, (int)x);
}
inline __host__ __device__ int3 make_int3(float x, float y){
    return make_int3((int)x, (int)y, 0);
}
inline __host__ __device__ int3 make_int3(float2 v){
    return make_int3((int)v.x, (int)v.y, 0);
}
inline __host__ __device__ int3 make_int3(float2 v, float x){
    return make_int3((int)v.x, (int)v.y, (int)x);
}
inline __host__ __device__ int3 make_int3(float x, float2 v){
    return make_int3((int)x, (int)v.x, (int)v.y);
}
inline __host__ __device__ int3 make_int3(float x, float y, float z){
    return make_int3((int)x, (int)y, (int)z);
}
inline __host__ __device__ int3 make_int3(float3 v){
    return make_int3((int)v.x, (int)v.y, (int)v.z);
}
inline __host__ __device__ int3 make_int3(float4 v){
    return make_int3((int)v.x, (int)v.y, (int)v.z);
}

inline __host__ __device__ int4 make_int4(uint x){
    return make_int4((int)x, (int)x, (int)x, (int)x);
}
inline __host__ __device__ int4 make_int4(uint x, uint y){
    return make_int4((int)x, (int)y, 0, 0);
}
inline __host__ __device__ int4 make_int4(uint2 v){
    return make_int4((int)v.x, (int)v.y, 0, 0);
}
inline __host__ __device__ int4 make_int4(uint x, uint y, uint z){
    return make_int4((int)x, (int)y, (int)z, 0);
}
inline __host__ __device__ int4 make_int4(uint2 x, uint y){
    return make_int4((int)x.x, (int)x.y, (int)y, 0);
}
inline __host__ __device__ int4 make_int4(uint x, uint2 y){
    return make_int4((int)x, (int)y.x, (int)y.y, 0);
}
inline __host__ __device__ int4 make_int4(uint3 v){
    return make_int4((int)v.x, (int)v.y, (int)v.z, 0);
}
inline __host__ __device__ int4 make_int4(uint x, uint y, uint z, uint w){
    return make_int4((int)x, (int)y, (int)z, (int)w);
}
inline __host__ __device__ int4 make_int4(uint x, uint y, uint2 z){
    return make_int4((int)x, (int)y, (int)z.x, (int)z.y);
}
inline __host__ __device__ int4 make_int4(uint x, uint2 y, uint z){
    return make_int4((int)x, (int)y.x, (int)y.y, (int)z);
}
inline __host__ __device__ int4 make_int4(uint2 x, uint y, uint z){
    return make_int4((int)x.x, (int)x.y, (int)y, (int)z);
}
inline __host__ __device__ int4 make_int4(uint2 x, uint2 y){
    return make_int4((int)x.x, (int)x.y, (int)y.x, (int)y.y);
}
inline __host__ __device__ int4 make_int4(uint3 v, uint w){
    return make_int4((int)v.x, (int)v.y, (int)v.z, (int)w);
}
inline __host__ __device__ int4 make_int4(uint x, uint3 v){
    return make_int4((int)x, (int)v.x, (int)v.y, (int)v.z);
}
inline __host__ __device__ int4 make_int4(uint4 v){
    return make_int4((int)v.x, (int)v.y, (int)v.z, (int)v.w);
}

inline __host__ __device__ int4 make_int4(float x){
    return make_int4((int)x, (int)x, (int)x, (int)x);
}
inline __host__ __device__ int4 make_int4(float x, float y){
    return make_int4((int)x, (int)y, 0, 0);
}
inline __host__ __device__ int4 make_int4(float2 v){
    return make_int4((int)v.x, (int)v.y, 0, 0);
}
inline __host__ __device__ int4 make_int4(float x, float y, float z){
    return make_int4((int)x, (int)y, (int)z, 0);
}
inline __host__ __device__ int4 make_int4(float2 x, float y){
    return make_int4((int)x.x, (int)x.y, (int)y, 0);
}
inline __host__ __device__ int4 make_int4(float x, float2 y){
    return make_int4((int)x, (int)y.x, (int)y.y, 0);
}
inline __host__ __device__ int4 make_int4(float3 v){
    return make_int4((int)v.x, (int)v.y, (int)v.z, 0);
}
inline __host__ __device__ int4 make_int4(float x, float y, float z, float w){
    return make_int4((int)x, (int)y, (int)z, (int)w);
}
inline __host__ __device__ int4 make_int4(float x, float y, float2 z){
    return make_int4((int)x, (int)y, (int)z.x, (int)z.y);
}
inline __host__ __device__ int4 make_int4(float x, float2 y, float z){
    return make_int4((int)x, (int)y.x, (int)y.y, (int)z);
}
inline __host__ __device__ int4 make_int4(float2 x, float y, float z){
    return make_int4((int)x.x, (int)x.y, (int)y, (int)z);
}
inline __host__ __device__ int4 make_int4(float2 x, float2 y){
    return make_int4((int)x.x, (int)x.y, (int)y.x, (int)y.y);
}
inline __host__ __device__ int4 make_int4(float3 v, float w){
    return make_int4((int)v.x, (int)v.y, (int)v.z, (int)w);
}
inline __host__ __device__ int4 make_int4(float x, float3 v){
    return make_int4((int)x, (int)v.x, (int)v.y, (int)v.z);
}
inline __host__ __device__ int4 make_int4(float4 v){
    return make_int4((int)v.x, (int)v.y, (int)v.z, (int)v.w);
}



inline __host__ __device__ longlong3 make_longlong3(int3 v){
    return make_longlong3((long long)v.x, (long long)v.y, (long long)v.z);
}
inline __host__ __device__ longlong3 make_longlong3(uint3 v){
    return make_longlong3((long long)v.x, (long long)v.y, (long long)v.z);
}






inline __host__ __device__ uchar4 make_uchar4(float x){
    return make_uchar4((uchar)x, (uchar)x, (uchar)x, (uchar)x);
}

inline __host__ __device__ uchar4 make_uchar4(float4 v){
    return make_uchar4((uchar)v.x, (uchar)v.y, (uchar)v.z, (uchar)v.w);
}




inline __host__ __device__ uint2 make_uint2(int x){
    return make_uint2((uint)x, (uint)x);
}
inline __host__ __device__ uint2 make_uint2(int x, int y){
    return make_uint2((uint)x, (uint)y);
}
inline __host__ __device__ uint2 make_uint2(int2 v){
    return make_uint2((uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint2 make_uint2(int3 v){
    return make_uint2((uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint2 make_uint2(int4 v){
    return make_uint2((uint)v.x, (uint)v.y);
}

inline __host__ __device__ uint2 make_uint2(float x){
    return make_uint2((uint)x, (uint)x);
}
inline __host__ __device__ uint2 make_uint2(float x, float y){
    return make_uint2((uint)x, (uint)y);
}
inline __host__ __device__ uint2 make_uint2(float2 v){
    return make_uint2((uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint2 make_uint2(float3 v){
    return make_uint2((uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint2 make_uint2(float4 v){
    return make_uint2((uint)v.x, (uint)v.y);
}


inline __host__ __device__ uint2 make_uint2(double x){
    return make_uint2((uint)x, (uint)x);
}
inline __host__ __device__ uint2 make_uint2(double x, double y){
    return make_uint2((uint)x, (uint)y);
}
inline __host__ __device__ uint2 make_uint2(double2 v){
    return make_uint2((uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint2 make_uint2(double3 v){
    return make_uint2((uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint2 make_uint2(double4 v){
    return make_uint2((uint)v.x, (uint)v.y);
}


inline __host__ __device__ uint3 make_uint3(int x){
    return make_uint3((uint)x, (uint)x, (uint)x);
}
inline __host__ __device__ uint3 make_uint3(int x, int y){
    return make_uint3((uint)x, (uint)y, 0u);
}
inline __host__ __device__ uint3 make_uint3(int2 v){
    return make_uint3((uint)v.x, (uint)v.y, 0u);
}
inline __host__ __device__ uint3 make_uint3(int2 v, int x){
    return make_uint3((uint)v.x, (uint)v.y, (uint)x);
}
inline __host__ __device__ uint3 make_uint3(int x, int2 v){
    return make_uint3((uint)x, (uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint3 make_uint3(int x, int y, int z){
    return make_uint3((uint)x, (uint)y, (uint)z);
}
inline __host__ __device__ uint3 make_uint3(int3 v){
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}
inline __host__ __device__ uint3 make_uint3(int4 v){
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}

inline __host__ __device__ uint3 make_uint3(float x){
    return make_uint3((uint)x, (uint)x, (uint)x);
}
inline __host__ __device__ uint3 make_uint3(float x, float y){
    return make_uint3((uint)x, (uint)y, 0u);
}
inline __host__ __device__ uint3 make_uint3(float2 v){
    return make_uint3((uint)v.x, (uint)v.y, 0u);
}
inline __host__ __device__ uint3 make_uint3(float2 v, float x){
    return make_uint3((uint)v.x, (uint)v.y, (uint)x);
}
inline __host__ __device__ uint3 make_uint3(float x, float2 v){
    return make_uint3((uint)x, (uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint3 make_uint3(float x, float y, float z){
    return make_uint3((uint)x, (uint)y, (uint)z);
}
inline __host__ __device__ uint3 make_uint3(float3 v){
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}
inline __host__ __device__ uint3 make_uint3(float4 v){
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}

inline __host__ __device__ uint3 make_uint3(double x){
    return make_uint3((uint)x, (uint)x, (uint)x);
}
inline __host__ __device__ uint3 make_uint3(double x, double y){
    return make_uint3((uint)x, (uint)y, 0u);
}
inline __host__ __device__ uint3 make_uint3(double2 v){
    return make_uint3((uint)v.x, (uint)v.y, 0u);
}
inline __host__ __device__ uint3 make_uint3(double2 v, double x){
    return make_uint3((uint)v.x, (uint)v.y, (uint)x);
}
inline __host__ __device__ uint3 make_uint3(double x, double2 v){
    return make_uint3((uint)x, (uint)v.x, (uint)v.y);
}
inline __host__ __device__ uint3 make_uint3(double x, double y, double z){
    return make_uint3((uint)x, (uint)y, (uint)z);
}
inline __host__ __device__ uint3 make_uint3(double3 v){
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}
inline __host__ __device__ uint3 make_uint3(double4 v){
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}

inline __host__ __device__ uint4 make_uint4(int x){
    return make_uint4((uint)x, (uint)x, (uint)x, (uint)x);
}
inline __host__ __device__ uint4 make_uint4(int x, int y){
    return make_uint4((uint)x, (uint)y, 0u, 0u);
}
inline __host__ __device__ uint4 make_uint4(int2 v){
    return make_uint4((uint)v.x, (uint)v.y, 0u, 0u);
}
inline __host__ __device__ uint4 make_uint4(int x, int y, int z){
    return make_uint4((uint)x, (uint)y, (uint)z, 0u);
}
inline __host__ __device__ uint4 make_uint4(int2 x, int y){
    return make_uint4((uint)x.x, (uint)x.y, (uint)y, 0u);
}
inline __host__ __device__ uint4 make_uint4(int x, int2 y){
    return make_uint4((uint)x, (uint)y.x, (uint)y.y, 0u);
}
inline __host__ __device__ uint4 make_uint4(int3 v){
    return make_uint4((uint)v.x, (uint)v.y, (uint)v.z, 0u);
}
inline __host__ __device__ uint4 make_uint4(int x, int y, int z, int w){
    return make_uint4((uint)x, (uint)y, (uint)z, (uint)w);
}
inline __host__ __device__ uint4 make_uint4(int x, int y, int2 z){
    return make_uint4((uint)x, (uint)y, (uint)z.x, (uint)z.y);
}
inline __host__ __device__ uint4 make_uint4(int x, int2 y, int z){
    return make_uint4((uint)x, (uint)y.x, (uint)y.y, (uint)z);
}
inline __host__ __device__ uint4 make_uint4(int2 x, int y, int z){
    return make_uint4((uint)x.x, (uint)x.y, (uint)y, (uint)z);
}
inline __host__ __device__ uint4 make_uint4(int2 x, int2 y){
    return make_uint4((uint)x.x, (uint)x.y, (uint)y.x, (uint)y.y);
}
inline __host__ __device__ uint4 make_uint4(int3 v, int w){
    return make_uint4((uint)v.x, (uint)v.y, (uint)v.z, (uint)w);
}
inline __host__ __device__ uint4 make_uint4(int x, int3 v){
    return make_uint4((uint)x, (uint)v.x, (uint)v.y, (uint)v.z);
}
inline __host__ __device__ uint4 make_uint4(int4 v){
    return make_uint4((uint)v.x, (uint)v.y, (uint)v.z, (uint)v.w);
}

inline __host__ __device__ uint4 make_uint4(float x){
    return make_uint4((uint)x, (uint)x, (uint)x, (uint)x);
}
inline __host__ __device__ uint4 make_uint4(float x, float y){
    return make_uint4((uint)x, (uint)y, 0u, 0u);
}
inline __host__ __device__ uint4 make_uint4(float2 v){
    return make_uint4((uint)v.x, (uint)v.y, 0u, 0u);
}
inline __host__ __device__ uint4 make_uint4(float x, float y, float z){
    return make_uint4((uint)x, (uint)y, (uint)z, 0u);
}
inline __host__ __device__ uint4 make_uint4(float2 x, float y){
    return make_uint4((uint)x.x, (uint)x.y, (uint)y, 0u);
}
inline __host__ __device__ uint4 make_uint4(float x, float2 y){
    return make_uint4((uint)x, (uint)y.x, (uint)y.y, 0u);
}
inline __host__ __device__ uint4 make_uint4(float3 v){
    return make_uint4((uint)v.x, (uint)v.y, (uint)v.z, 0u);
}
inline __host__ __device__ uint4 make_uint4(float x, float y, float z, float w){
    return make_uint4((uint)x, (uint)y, (uint)z, (uint)w);
}
inline __host__ __device__ uint4 make_uint4(float x, float y, float2 z){
    return make_uint4((uint)x, (uint)y, (uint)z.x, (uint)z.y);
}
inline __host__ __device__ uint4 make_uint4(float x, float2 y, float z){
    return make_uint4((uint)x, (uint)y.x, (uint)y.y, (uint)z);
}
inline __host__ __device__ uint4 make_uint4(float2 x, float y, float z){
    return make_uint4((uint)x.x, (uint)x.y, (uint)y, (uint)z);
}
inline __host__ __device__ uint4 make_uint4(float2 x, float2 y){
    return make_uint4((uint)x.x, (uint)x.y, (uint)y.x, (uint)y.y);
}
inline __host__ __device__ uint4 make_uint4(float3 v, float w){
    return make_uint4((uint)v.x, (uint)v.y, (uint)v.z, (uint)w);
}
inline __host__ __device__ uint4 make_uint4(float x, float3 v){
    return make_uint4((uint)x, (uint)v.x, (uint)v.y, (uint)v.z);
}
inline __host__ __device__ uint4 make_uint4(float4 v){
    return make_uint4((uint)v.x, (uint)v.y, (uint)v.z, (uint)v.w);
}


inline __host__ __device__ uint3 make_uint3(longlong3 v){
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}




inline __host__ __device__ ulonglong3 make_ulonglong3(uint3 v){
    return make_ulonglong3((ulonglong)v.x, (ulonglong)v.y, (ulonglong)v.z);
}










#endif