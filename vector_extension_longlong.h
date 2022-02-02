#ifndef VECTOR_EXTENSION_LONGLONG_H
#define VECTOR_EXTENSION_LONGLONG_H

#include "vector_extension_utils.h"


/////////////////////////////////////
// +
/////////////////////////////////////


inline __host__ __device__ longlong2 operator+(long long x, longlong2 y){
    return make_longlong2(x + y.x, x + y.y);
}
inline __host__ __device__ longlong2 operator+(longlong2 x, long long y){
    return make_longlong2(x.x + y, x.y + y);
}
inline __host__ __device__ longlong2 operator+(longlong2 x, longlong2 y){
    return make_longlong2(x.x + y.x, x.y + y.y);
}
inline __host__ __device__ longlong3 operator+(longlong3 x, long long y){
    return make_longlong3(x.x + y, x.y + y, x.z + y);
}
inline __host__ __device__ longlong3 operator+(long long x, longlong3 y){
    return make_longlong3(x + y.x, x + y.y, x + y.z);
}
inline __host__ __device__ longlong3 operator+(longlong3 x, longlong2 y){
    return make_longlong3(x.x + y.x, x.y + y.y, x.z);
}
inline __host__ __device__ longlong3 operator+(longlong2 x, longlong3 y){
    return make_longlong3(x.x + y.x, x.y + y.y, y.z);
}
inline __host__ __device__ longlong3 operator+(longlong3 x, longlong3 y){
    return make_longlong3(x.x + y.x, x.y + y.y, x.z + y.z);
}
inline __host__ __device__ longlong4 operator+(longlong4 x, long long y){
    return make_longlong4(x.x + y, x.y + y, x.z + y, x.w + y);
}
inline __host__ __device__ longlong4 operator+(long long x, longlong4 y){
    return make_longlong4(x + y.x, x + y.y, x + y.z, x + y.w);
}
inline __host__ __device__ longlong4 operator+(longlong4 x, longlong2 y){
    return make_longlong4(x.x + y.x, x.y + y.y, x.z, x.w);
}
inline __host__ __device__ longlong4 operator+(longlong2 x, longlong4 y){
    return make_longlong4(x.x + y.x, x.y + y.y, y.z, y.w);
}
inline __host__ __device__ longlong4 operator+(longlong4 x, longlong3 y){
    return make_longlong4(x.x + y.x, x.y + y.y, x.z + y.z, x.w);
}
inline __host__ __device__ longlong4 operator+(longlong3 x, longlong4 y){
    return make_longlong4(x.x + y.x, x.y + y.y, x.z + y.z, y.w);
}
inline __host__ __device__ longlong4 operator+(longlong4 x, longlong4 y){
    return make_longlong4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
}


/////////////////////////////////////
// -
/////////////////////////////////////


inline __host__ __device__ longlong2 operator-(long long x, longlong2 y){
    return make_longlong2(x - y.x, x - y.y);
}
inline __host__ __device__ longlong2 operator-(longlong2 x, long long y){
    return make_longlong2(x.x - y, x.y - y);
}
inline __host__ __device__ longlong2 operator-(longlong2 x, longlong2 y){
    return make_longlong2(x.x - y.x, x.y - y.y);
}
inline __host__ __device__ longlong3 operator-(long long x, longlong3 y){
    return make_longlong3(x - y.x, x - y.y, x - y.z);
}
inline __host__ __device__ longlong3 operator-(longlong3 x, long long y){
    return make_longlong3(x.x - y, x.y - y, x.z - y);
}
inline __host__ __device__ longlong3 operator-(longlong3 x, longlong2 y){
    return make_longlong3(x.x - y.x, x.y - y.y, x.z);
}
inline __host__ __device__ longlong3 operator-(longlong2 x, longlong3 y){
    return make_longlong3(x.x - y.x, x.y - y.y, y.z);
}
inline __host__ __device__ longlong3 operator-(longlong3 x, longlong3 y){
    return make_longlong3(x.x - y.x, x.y - y.y, x.z - y.z);
}
inline __host__ __device__ longlong4 operator-(long long x, longlong4 y){
    return make_longlong4(x - y.x, x - y.y, x - y.z, x - y.w);
}
inline __host__ __device__ longlong4 operator-(longlong4 x, long long y){
    return make_longlong4(x.x - y, x.y - y, x.z - y, x.w - y);
}
inline __host__ __device__ longlong4 operator-(longlong4 x, longlong2 y){
    return make_longlong4(x.x - y.x, x.y - y.y, x.z, x.w);
}
inline __host__ __device__ longlong4 operator-(longlong2 x, longlong4 y){
    return make_longlong4(x.x - y.x, x.y - y.y, y.z, y.w);
}
inline __host__ __device__ longlong4 operator-(longlong4 x, longlong3 y){
    return make_longlong4(x.x - y.x, x.y - y.y, x.z - y.z, x.w);
}
inline __host__ __device__ longlong4 operator-(longlong3 x, longlong4 y){
    return make_longlong4(x.x - y.x, x.y - y.y, x.z - y.z, y.w);
}
inline __host__ __device__ longlong4 operator-(longlong4 x, longlong4 y){
    return make_longlong4(x.x - y.x, x.y - y.y, x.z - y.z, x.z - y.z);
}


/////////////////////////////////////
// *
/////////////////////////////////////


inline __host__ __device__ longlong2 operator*(long long x, longlong2 y){
    return make_longlong2(x*y.x, x*y.y);
}
inline __host__ __device__ longlong2 operator*(longlong2 x, long long y){
    return make_longlong2(x.x*y, x.y*y);
}
inline __host__ __device__ longlong2 operator*(longlong2 x, longlong2 y){
    return make_longlong2(x.x*y.x, x.y*y.y);
}
inline __host__ __device__ longlong3 operator*(long long x, longlong3 y){
    return make_longlong3(x*y.x, x*y.y, x*y.z);
}
inline __host__ __device__ longlong3 operator*(longlong3 x, long long y){
    return make_longlong3(x.x*y, x.y*y, x.z*y);
}
inline __host__ __device__ longlong3 operator*(longlong3 x, longlong2 y){
    return make_longlong3(x.x*y.x, x.y*y.y, x.z);
}
inline __host__ __device__ longlong3 operator*(longlong2 x, longlong3 y){
    return make_longlong3(x.x*y.x, x.y*y.y, y.z);
}
inline __host__ __device__ longlong3 operator*(longlong3 x, longlong3 y){
    return make_longlong3(x.x*y.x, x.y*y.y, x.z*y.z);
}
inline __host__ __device__ longlong4 operator*(long long x, longlong4 y){
    return make_longlong4(x*y.x, x*y.y, x*y.z, x*y.w);
}
inline __host__ __device__ longlong4 operator*(longlong4 x, long long y){
    return make_longlong4(x.x*y, x.y*y, x.z*y, x.w*y);
}
inline __host__ __device__ longlong4 operator*(longlong4 x, longlong2 y){
    return make_longlong4(x.x*y.x, x.y*y.y, x.z, x.w);
}
inline __host__ __device__ longlong4 operator*(longlong2 x, longlong4 y){
    return make_longlong4(x.x*y.x, x.y*y.y, y.z, y.w);
}
inline __host__ __device__ longlong4 operator*(longlong4 x, longlong3 y){
    return make_longlong4(x.x*y.x, x.y*y.y, x.z*y.z, x.w);
}
inline __host__ __device__ longlong4 operator*(longlong3 x, longlong4 y){
    return make_longlong4(x.x*y.x, x.y*y.y, x.z*y.z, y.w);
}
inline __host__ __device__ longlong4 operator*(longlong4 x, longlong4 y){
    return make_longlong4(x.x*y.x, x.y*y.y, x.z*y.z, x.z*y.z);
}


/////////////////////////////////////
// /
/////////////////////////////////////


inline __host__ __device__ longlong2 operator/(long long x, longlong2 y){
    return make_longlong2(x/y.x, x/y.y);
}
inline __host__ __device__ longlong2 operator/(longlong2 x, long long y){
    return make_longlong2(x.x/y, x.y/y);
}
inline __host__ __device__ longlong2 operator/(longlong2 x, longlong2 y){
    return make_longlong2(x.x/y.x, x.y/y.y);
}
inline __host__ __device__ longlong3 operator/(long long x, longlong3 y){
    return make_longlong3(x/y.x, x/y.y, x/y.z);
}
inline __host__ __device__ longlong3 operator/(longlong3 x, long long y){
    return make_longlong3(x.x/y, x.y/y, x.z/y);
}
inline __host__ __device__ longlong3 operator/(longlong3 x, longlong2 y){
    return make_longlong3(x.x/y.x, x.y/y.y, x.z);
}
inline __host__ __device__ longlong3 operator/(longlong2 x, longlong3 y){
    return make_longlong3(x.x/y.x, x.y/y.y, y.z);
}
inline __host__ __device__ longlong3 operator/(longlong3 x, longlong3 y){
    return make_longlong3(x.x/y.x, x.y/y.y, x.z/y.z);
}
inline __host__ __device__ longlong4 operator/(long long x, longlong4 y){
    return make_longlong4(x/y.x, x/y.y, x/y.z, x/y.w);
}
inline __host__ __device__ longlong4 operator/(longlong4 x, long long y){
    return make_longlong4(x.x/y, x.y/y, x.z/y, x.w/y);
}
inline __host__ __device__ longlong4 operator/(longlong4 x, longlong2 y){
    return make_longlong4(x.x/y.x, x.y/y.y, x.z, x.w);
}
inline __host__ __device__ longlong4 operator/(longlong2 x, longlong4 y){
    return make_longlong4(x.x/y.x, x.y/y.y, y.z, y.w);
}
inline __host__ __device__ longlong4 operator/(longlong4 x, longlong3 y){
    return make_longlong4(x.x/y.x, x.y/y.y, x.z/y.z, x.w);
}
inline __host__ __device__ longlong4 operator/(longlong3 x, longlong4 y){
    return make_longlong4(x.x/y.x, x.y/y.y, x.z/y.z, y.w);
}
inline __host__ __device__ longlong4 operator/(longlong4 x, longlong4 y){
    return make_longlong4(x.x/y.x, x.y/y.y, x.z/y.z, x.z/y.z);
}


////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ longlong2 min(longlong2 x, longlong2 y){
    return make_longlong2(min(x.x, y.x), min(x.y, y.y)) ;
}

inline __host__ __device__ longlong3 min(longlong3 x, longlong3 y){
    return make_longlong3(min(x.x, y.x), min(x.y, y.y), min(x.z, y.z)) ;
}

inline __host__ __device__ longlong4 min(longlong4 x, longlong4 y){
    return make_longlong4(min(x.x, y.x), min(x.y, y.y), min(x.z, y.z), min(x.w, y.w)) ;
}

inline __host__ __device__ long long min(longlong2 x){
    return min(x.x, x.y);
}
inline __host__ __device__ long long min(longlong3 x){
    return min(x.x, min(x.y, x.z));
}
inline __host__ __device__ long long min(longlong4 x){
     return min(x.x, min(x.y, min(x.z, x.w)));
}
////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ longlong2 max(longlong2 x, longlong2 y){
    return make_longlong2(max(x.x, y.x), max(x.y, y.y)) ;
}

inline __host__ __device__ longlong3 max(longlong3 x, longlong3 y){
    return make_longlong3(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z)) ;
}

inline __host__ __device__ longlong4 max(longlong4 x, longlong4 y){
    return make_longlong4(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w)) ;
}

inline __host__ __device__ long long max(longlong2 x){
    return max(x.x, x.y);
}
inline __host__ __device__ long long max(longlong3 x){
    return max(x.x, max(x.y, x.z));
}
inline __host__ __device__ long long max(longlong4 x){
     return max(x.x, max(x.y, max(x.z, x.w)));
}

////////////////////////////////////////////////////////////////////////////////
// sum
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ long long sum(longlong2 x){
    return x.x + x.y;
}
inline __host__ __device__ long long sum(longlong3 x){
    return x.x + x.y + x.z;
}
inline __host__ __device__ long long sum(longlong4 x){
    return x.x + x.y + x.z + x.w;
}

////////////////////////////////////////////////////////////////////////////////
// product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ long long prod(longlong2 x){
    return x.x * x.y;
}
inline __host__ __device__ long long prod(longlong3 x){
    return x.x * x.y * x.z;
}
inline __host__ __device__ long long prod(longlong4 x){
    return x.x * x.y * x.z * x.w;
}




#endif