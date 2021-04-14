#ifndef VECTOR_EXTENSION_UINT_H
#define VECTOR_EXTENSION_UINT_H

#include "vector_extension_utils.cuh"


/////////////////////////////////////
// make_vector
/////////////////////////////////////


inline __host__ __device__ uint2 make_uint2(uint x){
    return make_uint2(x, x);
}
inline __host__ __device__ uint2 make_uint2(uint2 v){
    return make_uint2(v.x, v.y);
}
inline __host__ __device__ uint2 make_uint2(uint3 v){
    return make_uint2(v.x, v.y);
}
inline __host__ __device__ uint2 make_uint2(uint4 v){
    return make_uint2(v.x, v.y);
}
inline __host__ __device__ uint3 make_uint3(uint x){
    return make_uint3(x, x, x);
}
inline __host__ __device__ uint3 make_uint3(uint x, uint y){
    return make_uint3(x, y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 v){
    return make_uint3(v.x, v.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 v, uint x){
    return make_uint3(v.x, v.y, x);
}
inline __host__ __device__ uint3 make_uint3(uint x, uint2 v){
    return make_uint3(x, v.x, v.y);
}
inline __host__ __device__ uint3 make_uint3(uint3 v){
    return make_uint3(v.x, v.y, v.z);
}
inline __host__ __device__ uint3 make_uint3(uint4 v){
    return make_uint3(v.x, v.y, v.z);
}
inline __host__ __device__ uint4 make_uint4(uint x){
    return make_uint4(x, x, x, x);
}
inline __host__ __device__ uint4 make_uint4(uint x, uint y){
    return make_uint4(x, y, 0, 0);
}
inline __host__ __device__ uint4 make_uint4(uint2 v){
    return make_uint4(v.x, v.y, 0, 0);
}
inline __host__ __device__ uint4 make_uint4(uint x, uint y, uint z){
    return make_uint4(x, y, z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint2 x, uint y){
    return make_uint4(x.x, x.y, y, 0);
}
inline __host__ __device__ uint4 make_uint4(uint x, uint2 y){
    return make_uint4(x, y.x, y.y, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 v){
    return make_uint4(v.x, v.y, v.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint x, uint y, uint2 z){
    return make_uint4(x, y, z.x, z.y);
}
inline __host__ __device__ uint4 make_uint4(uint x, uint2 y, uint z){
    return make_uint4(x, y.x, y.y, z);
}
inline __host__ __device__ uint4 make_uint4(uint2 x, uint y, uint z){
    return make_uint4(x.x, x.y, y, z);
}
inline __host__ __device__ uint4 make_uint4(uint2 x, uint2 y){
    return make_uint4(x.x, x.y, y.x, y.y);
}
inline __host__ __device__ uint4 make_uint4(uint3 v, uint w){
    return make_uint4(v.x, v.y, v.z, w);
}
inline __host__ __device__ uint4 make_uint4(uint x, uint3 v){
    return make_uint4(x, v.x, v.y, v.z);
}
inline __host__ __device__ uint4 make_uint4(uint4 v){
    return make_uint4(v.x, v.y, v.z, v.w);
}


////////////////////////////////////////////////////////////////////////////////
// +
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator+(uint x, uint2 y){
    return make_uint2(x + y.x, x + y.y);
}
inline __host__ __device__ uint2 operator+(uint2 x, uint y){
    return make_uint2(x.x + y, x.y + y);
}
inline __host__ __device__ uint2 operator+(uint2 x, uint2 y){
    return make_uint2(x.x + y.x, x.y + y.y);
}
inline __host__ __device__ uint3 operator+(uint3 x, uint y){
    return make_uint3(x.x + y, x.y + y, x.z + y);
}
inline __host__ __device__ uint3 operator+(uint3 x, uint2 y){
    return make_uint3(x.x + y.x, x.y + y.y, x.z);
}
inline __host__ __device__ uint3 operator+(uint2 x, uint3 y){
    return make_uint3(x.x + y.x, x.y + y.y, y.z);
}
inline __host__ __device__ uint3 operator+(uint3 x, uint3 y){
    return make_uint3(x.x + y.x, x.y + y.y, x.z + y.z);
}
inline __host__ __device__ uint4 operator+(uint4 x, uint y){
    return make_uint4(x.x + y, x.y + y, x.z + y);
}
inline __host__ __device__ uint4 operator+(uint x, uint4 y){
    return make_uint4(y.x + x, y.y + x, y.z + x);
}
inline __host__ __device__ uint4 operator+(uint4 x, uint2 y){
    return make_uint4(x.x + y.x, x.y + y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator+(uint2 x, uint4 y){
    return make_uint4(x.x + y.x, x.y + y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator+(uint4 x, uint3 y){
    return make_uint4(x.x + y.x, x.y + y.y, x.z + y.z, x.w);
}
inline __host__ __device__ uint4 operator+(uint3 x, uint4 y){
    return make_uint4(x.x + y.x, x.y + y.y, x.z + y.z, y.w);
}
inline __host__ __device__ uint4 operator+(uint4 x, uint4 y){
    return make_uint4(x.x + y.x, x.y + y.y, x.z + y.z);
}


////////////////////////////////////////////////////////////////////////////////
// -
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator-(uint x, uint2 y){
    return make_uint2(x - y.x, x - y.y);
}
inline __host__ __device__ uint2 operator-(uint2 x, uint y){
    return make_uint2(x.x - y, x.y - y);
}
inline __host__ __device__ uint2 operator-(uint2 x, uint2 y){
    return make_uint2(x.x - y.x, x.y - y.y);
}

inline __host__ __device__ uint3 operator-(uint3 x, uint y){
    return make_uint3(x.x - y, x.y - y, x.z - y);
}
inline __host__ __device__ uint3 operator-(uint3 x, uint2 y){
    return make_uint3(x.x - y.x, x.y - y.y, x.z);
}
inline __host__ __device__ uint3 operator-(uint2 x, uint3 y){
    return make_uint3(x.x - y.x, x.y - y.y, y.z);
}
inline __host__ __device__ uint3 operator-(uint3 x, uint3 y){
    return make_uint3(x.x - y.x, x.y - y.y, x.z - y.z);
}
inline __host__ __device__ uint4 operator-(uint4 x, uint y){
    return make_uint4(x.x - y, x.y - y, x.z - y);
}
inline __host__ __device__ uint4 operator-(uint x, uint4 y){
    return make_uint4(y.x - x, y.y - x, y.z - x);
}
inline __host__ __device__ uint4 operator-(uint4 x, uint2 y){
    return make_uint4(x.x - y.x, x.y - y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator-(uint2 x, uint4 y){
    return make_uint4(x.x - y.x, x.y - y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator-(uint4 x, uint3 y){
    return make_uint4(x.x - y.x, x.y - y.y, x.z - y.z, x.w);
}
inline __host__ __device__ uint4 operator-(uint3 x, uint4 y){
    return make_uint4(x.x - y.x, x.y - y.y, x.z - y.z, y.w);
}
inline __host__ __device__ uint4 operator-(uint4 x, uint4 y){
    return make_uint4(x.x - y.x, x.y - y.y, x.z - y.z);
}


////////////////////////////////////////////////////////////////////////////////
// *
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator*(uint x, uint2 y){
    return make_uint2(x*y.x, x*y.y);
}
inline __host__ __device__ uint2 operator*(uint2 x, uint y){
    return make_uint2(x.x*y, x.y*y);
}
inline __host__ __device__ uint2 operator*(uint2 x, uint2 y){
    return make_uint2(x.x*y.x, x.y*y.y);
}
inline __host__ __device__ uint3 operator*(uint3 x, uint y){
    return make_uint3(x.x*y, x.y*y, x.z*y);
}
inline __host__ __device__ uint3 operator*(uint3 x, uint2 y){
    return make_uint3(x.x*y.x, x.y*y.y, x.z);
}
inline __host__ __device__ uint3 operator*(uint2 x, uint3 y){
    return make_uint3(x.x*y.x, x.y*y.y, y.z);
}
inline __host__ __device__ uint3 operator*(uint3 x, uint3 y){
    return make_uint3(x.x*y.x, x.y*y.y, x.z*y.z);
}
inline __host__ __device__ uint4 operator*(uint4 x, uint y){
    return make_uint4(x.x*y, x.y*y, x.z*y);
}
inline __host__ __device__ uint4 operator*(uint x, uint4 y){
    return make_uint4(y.x*x, y.y*x, y.z*x);
}
inline __host__ __device__ uint4 operator*(uint4 x, uint2 y){
    return make_uint4(x.x*y.x, x.y*y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator*(uint2 x, uint4 y){
    return make_uint4(x.x*y.x, x.y*y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator*(uint4 x, uint3 y){
    return make_uint4(x.x*y.x, x.y*y.y, x.z*y.z, x.w);
}
inline __host__ __device__ uint4 operator*(uint3 x, uint4 y){
    return make_uint4(x.x*y.x, x.y*y.y, x.z*y.z, y.w);
}
inline __host__ __device__ uint4 operator*(uint4 x, uint4 y){
    return make_uint4(x.x*y.x, x.y*y.y, x.z*y.z);
}


////////////////////////////////////////////////////////////////////////////////
// /
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator/(uint x, uint2 y){
    return make_uint2(x/y.x, x/y.y);
}
inline __host__ __device__ uint2 operator/(uint2 x, uint y){
    return make_uint2(x.x/y, x.y/y);
}
inline __host__ __device__ uint2 operator/(uint2 x, uint2 y){
    return make_uint2(x.x/y.x, x.y/y.y);
}
inline __host__ __device__ uint3 operator/(uint3 x, uint y){
    return make_uint3(x.x/y, x.y/y, x.z/y);
}
inline __host__ __device__ uint3 operator/(uint3 x, uint2 y){
    return make_uint3(x.x/y.x, x.y/y.y, x.z);
}
inline __host__ __device__ uint3 operator/(uint2 x, uint3 y){
    return make_uint3(x.x/y.x, x.y/y.y, y.z);
}
inline __host__ __device__ uint3 operator/(uint3 x, uint3 y){
    return make_uint3(x.x/y.x, x.y/y.y, x.z/y.z);
}
inline __host__ __device__ uint4 operator/(uint4 x, uint y){
    return make_uint4(x.x/y, x.y/y, x.z/y);
}
inline __host__ __device__ uint4 operator/(uint x, uint4 y){
    return make_uint4(y.x/x, y.y/x, y.z/x);
}
inline __host__ __device__ uint4 operator/(uint4 x, uint2 y){
    return make_uint4(x.x/y.x, x.y/y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator/(uint2 x, uint4 y){
    return make_uint4(x.x/y.x, x.y/y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator/(uint4 x, uint3 y){
    return make_uint4(x.x/y.x, x.y/y.y, x.z/y.z, x.w);
}
inline __host__ __device__ uint4 operator/(uint3 x, uint4 y){
    return make_uint4(x.x/y.x, x.y/y.y, x.z/y.z, y.w);
}
inline __host__ __device__ uint4 operator/(uint4 x, uint4 y){
    return make_uint4(x.x/y.x, x.y/y.y, x.z/y.z);
}


////////////////////////////////////////////////////////////////////////////////
// |
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator|(uint x, uint2 y){
    return make_uint2(x|y.x, x|y.y);
}
inline __host__ __device__ uint2 operator|(uint2 x, uint y){
    return make_uint2(x.x|y, x.y|y);
}
inline __host__ __device__ uint2 operator|(uint2 x, uint2 y){
    return make_uint2(x.x|y.x, x.y|y.y);
}
inline __host__ __device__ uint3 operator|(uint3 x, uint y){
    return make_uint3(x.x|y, x.y|y, x.z|y);
}
inline __host__ __device__ uint3 operator|(uint3 x, uint2 y){
    return make_uint3(x.x|y.x, x.y|y.y, x.z);
}
inline __host__ __device__ uint3 operator|(uint2 x, uint3 y){
    return make_uint3(x.x|y.x, x.y|y.y, y.z);
}
inline __host__ __device__ uint3 operator|(uint3 x, uint3 y){
    return make_uint3(x.x|y.x, x.y|y.y, x.z|y.z);
}
inline __host__ __device__ uint4 operator|(uint4 x, uint y){
    return make_uint4(x.x|y, x.y|y, x.z|y);
}
inline __host__ __device__ uint4 operator|(uint x, uint4 y){
    return make_uint4(y.x|x, y.y|x, y.z|x);
}
inline __host__ __device__ uint4 operator|(uint4 x, uint2 y){
    return make_uint4(x.x|y.x, x.y|y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator|(uint2 x, uint4 y){
    return make_uint4(x.x|y.x, x.y|y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator|(uint4 x, uint3 y){
    return make_uint4(x.x|y.x, x.y|y.y, x.z|y.z, x.w);
}
inline __host__ __device__ uint4 operator|(uint3 x, uint4 y){
    return make_uint4(x.x|y.x, x.y|y.y, x.z|y.z, y.w);
}
inline __host__ __device__ uint4 operator|(uint4 x, uint4 y){
    return make_uint4(x.x|y.x, x.y|y.y, x.z|y.z);
}


////////////////////////////////////////////////////////////////////////////////
// ^
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator^(uint x, uint2 y){
    return make_uint2(x^y.x, x^y.y);
}
inline __host__ __device__ uint2 operator^(uint2 x, uint y){
    return make_uint2(x.x^y, x.y^y);
}
inline __host__ __device__ uint2 operator^(uint2 x, uint2 y){
    return make_uint2(x.x^y.x, x.y^y.y);
}
inline __host__ __device__ uint3 operator^(uint3 x, uint y){
    return make_uint3(x.x^y, x.y^y, x.z^y);
}
inline __host__ __device__ uint3 operator^(uint3 x, uint2 y){
    return make_uint3(x.x^y.x, x.y^y.y, x.z);
}
inline __host__ __device__ uint3 operator^(uint2 x, uint3 y){
    return make_uint3(x.x^y.x, x.y^y.y, y.z);
}
inline __host__ __device__ uint3 operator^(uint3 x, uint3 y){
    return make_uint3(x.x^y.x, x.y^y.y, x.z^y.z);
}
inline __host__ __device__ uint4 operator^(uint4 x, uint y){
    return make_uint4(x.x^y, x.y^y, x.z^y);
}
inline __host__ __device__ uint4 operator^(uint x, uint4 y){
    return make_uint4(y.x^x, y.y^x, y.z^x);
}
inline __host__ __device__ uint4 operator^(uint4 x, uint2 y){
    return make_uint4(x.x^y.x, x.y^y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator^(uint2 x, uint4 y){
    return make_uint4(x.x^y.x, x.y^y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator^(uint4 x, uint3 y){
    return make_uint4(x.x^y.x, x.y^y.y, x.z^y.z, x.w);
}
inline __host__ __device__ uint4 operator^(uint3 x, uint4 y){
    return make_uint4(x.x^y.x, x.y^y.y, x.z^y.z, y.w);
}
inline __host__ __device__ uint4 operator^(uint4 x, uint4 y){
    return make_uint4(x.x^y.x, x.y^y.y, x.z^y.z);
}


////////////////////////////////////////////////////////////////////////////////
// ~
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator~(uint2 v){
    return make_uint2(~v.x, ~v.y);
}
inline __host__ __device__ uint3 operator~(uint3 v){
    return make_uint3(~v.x, ~v.y, ~v.z);
}
inline __host__ __device__ uint4 operator~(uint4 v){
    return make_uint4(~v.x, ~v.y, ~v.z, ~v.w);
}


////////////////////////////////////////////////////////////////////////////////
// <<
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator<<(uint x, uint2 y){
    return make_uint2(x<<y.x, x<<y.y);
}
inline __host__ __device__ uint2 operator<<(uint2 x, uint y){
    return make_uint2(x.x<<y, x.y<<y);
}
inline __host__ __device__ uint2 operator<<(uint2 x, uint2 y){
    return make_uint2(x.x<<y.x, x.y<<y.y);
}
inline __host__ __device__ uint3 operator<<(uint3 x, uint y){
    return make_uint3(x.x<<y, x.y<<y, x.z<<y);
}
inline __host__ __device__ uint3 operator<<(uint3 x, uint2 y){
    return make_uint3(x.x<<y.x, x.y<<y.y, x.z);
}
inline __host__ __device__ uint3 operator<<(uint2 x, uint3 y){
    return make_uint3(x.x<<y.x, x.y<<y.y, y.z);
}
inline __host__ __device__ uint3 operator<<(uint3 x, uint3 y){
    return make_uint3(x.x<<y.x, x.y<<y.y, x.z<<y.z);
}
inline __host__ __device__ uint4 operator<<(uint4 x, uint y){
    return make_uint4(x.x<<y, x.y<<y, x.z<<y);
}
inline __host__ __device__ uint4 operator<<(uint x, uint4 y){
    return make_uint4(y.x<<x, y.y<<x, y.z<<x);
}
inline __host__ __device__ uint4 operator<<(uint4 x, uint2 y){
    return make_uint4(x.x<<y.x, x.y<<y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator<<(uint2 x, uint4 y){
    return make_uint4(x.x<<y.x, x.y<<y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator<<(uint4 x, uint3 y){
    return make_uint4(x.x<<y.x, x.y<<y.y, x.z<<y.z, x.w);
}
inline __host__ __device__ uint4 operator<<(uint3 x, uint4 y){
    return make_uint4(x.x<<y.x, x.y<<y.y, x.z<<y.z, y.w);
}
inline __host__ __device__ uint4 operator<<(uint4 x, uint4 y){
    return make_uint4(x.x<<y.x, x.y<<y.y, x.z<<y.z);
}


////////////////////////////////////////////////////////////////////////////////
// >>
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint2 operator>>(uint x, uint2 y){
    return make_uint2(x>>y.x, x>>y.y);
}
inline __host__ __device__ uint2 operator>>(uint2 x, uint y){
    return make_uint2(x.x>>y, x.y>>y);
}
inline __host__ __device__ uint2 operator>>(uint2 x, uint2 y){
    return make_uint2(x.x>>y.x, x.y>>y.y);
}
inline __host__ __device__ uint3 operator>>(uint3 x, uint y){
    return make_uint3(x.x>>y, x.y>>y, x.z>>y);
}
inline __host__ __device__ uint3 operator>>(uint3 x, uint2 y){
    return make_uint3(x.x>>y.x, x.y>>y.y, x.z);
}
inline __host__ __device__ uint3 operator>>(uint2 x, uint3 y){
    return make_uint3(x.x>>y.x, x.y>>y.y, y.z);
}
inline __host__ __device__ uint3 operator>>(uint3 x, uint3 y){
    return make_uint3(x.x>>y.x, x.y>>y.y, x.z>>y.z);
}
inline __host__ __device__ uint4 operator>>(uint4 x, uint y){
    return make_uint4(x.x>>y, x.y>>y, x.z>>y);
}
inline __host__ __device__ uint4 operator>>(uint x, uint4 y){
    return make_uint4(y.x>>x, y.y>>x, y.z>>x);
}
inline __host__ __device__ uint4 operator>>(uint4 x, uint2 y){
    return make_uint4(x.x>>y.x, x.y>>y.y, x.z, x.w);
}
inline __host__ __device__ uint4 operator>>(uint2 x, uint4 y){
    return make_uint4(x.x>>y.x, x.y>>y.y, y.z, y.w);
}
inline __host__ __device__ uint4 operator>>(uint4 x, uint3 y){
    return make_uint4(x.x>>y.x, x.y>>y.y, x.z>>y.z, x.w);
}
inline __host__ __device__ uint4 operator>>(uint3 x, uint4 y){
    return make_uint4(x.x>>y.x, x.y>>y.y, x.z>>y.z, y.w);
}
inline __host__ __device__ uint4 operator>>(uint4 x, uint4 y){
    return make_uint4(x.x>>y.x, x.y>>y.y, x.z>>y.z);
}


////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////


inline  __host__ __device__ uint2 min(uint x, uint2 y){
    return make_uint2(min(x,y.x), min(x,y.y));
}
inline  __host__ __device__ uint2 min(uint2 x, uint y){
    return make_uint2(min(x.x,y), min(x.y,y));
}
inline  __host__ __device__ uint2 min(uint2 x, uint2 y){
    return make_uint2(min(x.x,y.x), min(x.y,y.y));
}
inline __host__ __device__ uint3 min(uint x, uint3 y){
    return make_uint3(min(x,y.x), min(x,y.y), min(x,y.z));
}
inline __host__ __device__ uint3 min(uint3 x, uint y){
    return make_uint3(min(x.x,y), min(x.y,y), min(x.z,y));
}
inline __host__ __device__ uint3 min(uint3 x, uint3 y){
    return make_uint3(min(x.x,y.x), min(x.y,y.y), min(x.z,y.z));
}
inline  __host__ __device__ uint4 min(uint x, uint4 y){
    return make_uint4(min(x,y.x), min(x,y.y), min(x,y.z), min(x,y.w));
}
inline  __host__ __device__ uint4 min(uint4 x, uint y){
    return make_uint4(min(x.x,y), min(x.y,y), min(x.z,y), min(x.w,y));
}
inline  __host__ __device__ uint4 min(uint4 x, uint4 y){
    return make_uint4(min(x.x,y.x), min(x.y,y.y), min(x.z,y.z), min(x.w,y.w));
}


////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////


inline  __host__ __device__ uint2 max(uint x, uint2 y){
    return make_uint2(max(x,y.x), max(x,y.y));
}
inline  __host__ __device__ uint2 max(uint2 x, uint y){
    return make_uint2(max(x.x,y), max(x.y,y));
}
inline  __host__ __device__ uint2 max(uint2 x, uint2 y){
    return make_uint2(max(x.x,y.x), max(x.y,y.y));
}
inline __host__ __device__ uint3 max(uint x, uint3 y){
    return make_uint3(max(x,y.x), max(x,y.y), max(x,y.z));
}
inline __host__ __device__ uint3 max(uint3 x, uint y){
    return make_uint3(max(x.x,y), max(x.y,y), max(x.z,y));
}
inline __host__ __device__ uint3 max(uint3 x, uint3 y){
    return make_uint3(max(x.x,y.x), max(x.y,y.y), max(x.z,y.z));
}
inline  __host__ __device__ uint4 max(uint x, uint4 y){
    return make_uint4(max(x,y.x), max(x,y.y), max(x,y.z), max(x,y.w));
}
inline  __host__ __device__ uint4 max(uint4 x, uint y){
    return make_uint4(max(x.x,y), max(x.y,y), max(x.z,y), max(x.w,y));
}
inline  __host__ __device__ uint4 max(uint4 x, uint4 y){
    return make_uint4(max(x.x,y.x), max(x.y,y.y), max(x.z,y.z), max(x.w,y.w));
}


////////////////////////////////////////////////////////////////////////////////
// dot
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint dot(uint2 x, uint2 y){
    return x.x*y.x + y.y*y.y;
}
inline __host__ __device__ uint dot(uint3 x, uint3 y){
    return x.x*y.x + x.y*y.y + x.z*y.z;
}
inline __host__ __device__ uint dot(uint4 x, uint4 y){
    return x.x*y.x + x.y*y.y + x.z*y.z + x.w*y.w;
}


////////////////////////////////////////////////////////////////////////////////
// sum
////////////////////////////////////////////////////////////////////////////////


inline __host__ __device__ uint sum(uint2 x){
    return x.x + x.y;
}
inline __host__ __device__ uint sum(uint3 x){
    return x.x + x.y + x.z;
}
inline __host__ __device__ uint sum(uint4 x){
    return x.x + x.y + x.z + x.w;
}


#endif