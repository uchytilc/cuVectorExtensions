#ifndef VECTOR_EXTENSION_INT_H
#define VECTOR_EXTENSION_INT_H

#include "vector_extension_utils.cuh"


/////////////////////////////////////
// make_vector
/////////////////////////////////////


inline __host__ __device__ int2 make_int2(int x){
    return make_int2(x, x);
}
inline __host__ __device__ int2 make_int2(int2 v){
    return make_int2(v.x, v.y);
}
inline __host__ __device__ int2 make_int2(int3 v){
    return make_int2(v.x, v.y);
}
inline __host__ __device__ int2 make_int2(int4 v){
    return make_int2(v.x, v.y);
}
inline __host__ __device__ int3 make_int3(int x){
    return make_int3(x, x, x);
}
inline __host__ __device__ int3 make_int3(int x, int y){
    return make_int3(x, y, 0);
}
inline __host__ __device__ int3 make_int3(int2 v){
    return make_int3(v.x, v.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 v, int x){
    return make_int3(v.x, v.y, x);
}
inline __host__ __device__ int3 make_int3(int x, int2 v){
    return make_int3(x, v.x, v.y);
}
inline __host__ __device__ int3 make_int3(int3 v){
    return make_int3(v.x, v.y, v.z);
}
inline __host__ __device__ int3 make_int3(int4 v){
    return make_int3(v.x, v.y, v.z);
}
inline __host__ __device__ int4 make_int4(int x){
    return make_int4(x, x, x, x);
}
inline __host__ __device__ int4 make_int4(int x, int y){
    return make_int4(x, y, 0, 0);
}
inline __host__ __device__ int4 make_int4(int2 v){
    return make_int4(v.x, v.y, 0, 0);
}
inline __host__ __device__ int4 make_int4(int x, int y, int z){
    return make_int4(x, y, z, 0);
}
inline __host__ __device__ int4 make_int4(int2 x, int y){
    return make_int4(x.x, x.y, y, 0);
}
inline __host__ __device__ int4 make_int4(int x, int2 y){
    return make_int4(x, y.x, y.y, 0);
}
inline __host__ __device__ int4 make_int4(int3 v){
    return make_int4(v.x, v.y, v.z, 0);
}
inline __host__ __device__ int4 make_int4(int x, int y, int2 z){
    return make_int4(x, y, z.x, z.y);
}
inline __host__ __device__ int4 make_int4(int x, int2 y, int z){
    return make_int4(x, y.x, y.y, z);
}
inline __host__ __device__ int4 make_int4(int2 x, int y, int z){
    return make_int4(x.x, x.y, y, z);
}
inline __host__ __device__ int4 make_int4(int2 x, int2 y){
    return make_int4(x.x, x.y, y.x, y.y);
}
inline __host__ __device__ int4 make_int4(int3 v, int w){
    return make_int4(v.x, v.y, v.z, w);
}
inline __host__ __device__ int4 make_int4(int x, int3 v){
    return make_int4(x, v.x, v.y, v.z);
}
inline __host__ __device__ int4 make_int4(int4 v){
    return make_int4(v.x, v.y, v.z, v.w);
}


/////////////////////////////////////
// negate
/////////////////////////////////////


inline __host__ __device__ int2 operator-(int2 v){ //const int3& v
    return make_int2(-v.x, -v.y);
}
inline __host__ __device__ int3 operator-(int3 v){
    return make_int3(-v.x, -v.y, -v.z);
}
inline __host__ __device__ int4 operator-(int4 v){
    return make_int4(-v.x, -v.y, -v.z, -v.w);
}


/////////////////////////////////////
// +
/////////////////////////////////////


inline __host__ __device__ int2 operator+(int x, int2 y){
    return make_int2(x + y.x, x + y.y);
}
inline __host__ __device__ int2 operator+(int2 x, int y){
    return make_int2(x.x + y, x.y + y);
}
inline __host__ __device__ int2 operator+(int2 x, int2 y){
    return make_int2(x.x + y.x, x.y + y.y);
}
inline __host__ __device__ int3 operator+(int3 x, int y){
    return make_int3(x.x + y, x.y + y, x.z + y);
}
inline __host__ __device__ int3 operator+(int x, int3 y){
    return make_int3(x + y.x, x + y.y, x + y.z);
}
inline __host__ __device__ int3 operator+(int3 x, int2 y){
    return make_int3(x.x + y.x, x.y + y.y, x.z);
}
inline __host__ __device__ int3 operator+(int2 x, int3 y){
    return make_int3(x.x + y.x, x.y + y.y, y.z);
}
inline __host__ __device__ int3 operator+(int3 x, int3 y){
    return make_int3(x.x + y.x, x.y + y.y, x.z + y.z);
}
inline __host__ __device__ int4 operator+(int4 x, int y){
    return make_int4(x.x + y, x.y + y, x.z + y, x.w + y);
}
inline __host__ __device__ int4 operator+(int x, int4 y){
    return make_int4(x + y.x, x + y.y, x + y.z, x + y.w);
}
inline __host__ __device__ int4 operator+(int4 x, int2 y){
    return make_int4(x.x + y.x, x.y + y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator+(int2 x, int4 y){
    return make_int4(x.x + y.x, x.y + y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator+(int4 x, int3 y){
    return make_int4(x.x + y.x, x.y + y.y, x.z + y.z, x.w);
}
inline __host__ __device__ int4 operator+(int3 x, int4 y){
    return make_int4(x.x + y.x, x.y + y.y, x.z + y.z, y.w);
}
inline __host__ __device__ int4 operator+(int4 x, int4 y){
    return make_int4(x.x + y.x, x.y + y.y, x.z + y.z);
}


/////////////////////////////////////
// -
/////////////////////////////////////


inline __host__ __device__ int2 operator-(int x, int2 y){
    return make_int2(x - y.x, x - y.y);
}
inline __host__ __device__ int2 operator-(int2 x, int y){
    return make_int2(x.x - y, x.y - y);
}
inline __host__ __device__ int2 operator-(int2 x, int2 y){
    return make_int2(x.x - y.x, x.y - y.y);
}
inline __host__ __device__ int3 operator-(int3 x, int y){
    return make_int3(x.x - y, x.y - y, x.z - y);
}
inline __host__ __device__ int3 operator-(int x, int3 y){
    return make_int3(x - y.x, x - y.y, x - y.z);
}
inline __host__ __device__ int3 operator-(int3 x, int2 y){
    return make_int3(x.x - y.x, x.y - y.y, x.z);
}
inline __host__ __device__ int3 operator-(int2 x, int3 y){
    return make_int3(x.x - y.x, x.y - y.y, y.z);
}
inline __host__ __device__ int3 operator-(int3 x, int3 y){
    return make_int3(x.x - y.x, x.y - y.y, x.z - y.z);
}
inline __host__ __device__ int4 operator-(int4 x, int y){
    return make_int4(x.x - y, x.y - y, x.z - y, x.w - y);
}
inline __host__ __device__ int4 operator-(int x, int4 y){
    return make_int4(x - y.x, x - y.y, x - y.z, x - y.w);
}
inline __host__ __device__ int4 operator-(int4 x, int2 y){
    return make_int4(x.x - y.x, x.y - y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator-(int2 x, int4 y){
    return make_int4(x.x - y.x, x.y - y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator-(int4 x, int3 y){
    return make_int4(x.x - y.x, x.y - y.y, x.z - y.z, x.w);
}
inline __host__ __device__ int4 operator-(int3 x, int4 y){
    return make_int4(x.x - y.x, x.y - y.y, x.z - y.z, y.w);
}
inline __host__ __device__ int4 operator-(int4 x, int4 y){
    return make_int4(x.x - y.x, x.y - y.y, x.z - y.z);
}


/////////////////////////////////////
// *
/////////////////////////////////////


inline __host__ __device__ int2 operator*(int x, int2 y){
    return make_int2(x*y.x, x*y.y);
}
inline __host__ __device__ int2 operator*(int2 x, int y){
    return make_int2(x.x*y, x.y*y);
}
inline __host__ __device__ int2 operator*(int2 x, int2 y){
    return make_int2(x.x*y.x, x.y*y.y);
}
inline __host__ __device__ int3 operator*(int3 x, int y){
    return make_int3(x.x*y, x.y*y, x.z*y);
}
inline __host__ __device__ int3 operator*(int x, int3 y){
    return make_int3(x*y.x, x*y.y, x*y.z);
}
inline __host__ __device__ int3 operator*(int3 x, int2 y){
    return make_int3(x.x*y.x, x.y*y.y, x.z);
}
inline __host__ __device__ int3 operator*(int2 x, int3 y){
    return make_int3(x.x*y.x, x.y*y.y, y.z);
}
inline __host__ __device__ int3 operator*(int3 x, int3 y){
    return make_int3(x.x*y.x, x.y*y.y, x.z*y.z);
}
inline __host__ __device__ int4 operator*(int4 x, int y){
    return make_int4(x.x*y, x.y*y, x.z*y, x.w*y);
}
inline __host__ __device__ int4 operator*(int x, int4 y){
    return make_int4(x*y.x, x*y.y, x*y.z, x*y.w);
}
inline __host__ __device__ int4 operator*(int4 x, int2 y){
    return make_int4(x.x*y.x, x.y*y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator*(int2 x, int4 y){
    return make_int4(x.x*y.x, x.y*y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator*(int4 x, int3 y){
    return make_int4(x.x*y.x, x.y*y.y, x.z*y.z, x.w);
}
inline __host__ __device__ int4 operator*(int3 x, int4 y){
    return make_int4(x.x*y.x, x.y*y.y, x.z*y.z, y.w);
}
inline __host__ __device__ int4 operator*(int4 x, int4 y){
    return make_int4(x.x*y.x, x.y*y.y, x.z*y.z);
}


/////////////////////////////////////
// /
/////////////////////////////////////


inline __host__ __device__ int2 operator/(int x, int2 y){
    return make_int2(x/y.x, x/y.y);
}
inline __host__ __device__ int2 operator/(int2 x, int y){
    return make_int2(x.x/y, x.y/y);
}
inline __host__ __device__ int2 operator/(int2 x, int2 y){
    return make_int2(x.x/y.x, x.y/y.y);
}
inline __host__ __device__ int3 operator/(int3 x, int y){
    return make_int3(x.x/y, x.y/y, x.z/y);
}
inline __host__ __device__ int3 operator/(int x, int3 y){
    return make_int3(x/y.x, x/y.y, x/y.z);
}
inline __host__ __device__ int3 operator/(int3 x, int2 y){
    return make_int3(x.x/y.x, x.y/y.y, x.z);
}
inline __host__ __device__ int3 operator/(int2 x, int3 y){
    return make_int3(x.x/y.x, x.y/y.y, y.z);
}
inline __host__ __device__ int3 operator/(int3 x, int3 y){
    return make_int3(x.x/y.x, x.y/y.y, x.z/y.z);
}
inline __host__ __device__ int4 operator/(int4 x, int y){
    return make_int4(x.x/y, x.y/y, x.z/y, x.w/y);
}
inline __host__ __device__ int4 operator/(int x, int4 y){
    return make_int4(x/y.x, x/y.y, x/y.z, x/y.w);
}
inline __host__ __device__ int4 operator/(int4 x, int2 y){
    return make_int4(x.x/y.x, x.y/y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator/(int2 x, int4 y){
    return make_int4(x.x/y.x, x.y/y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator/(int4 x, int3 y){
    return make_int4(x.x/y.x, x.y/y.y, x.z/y.z, x.w);
}
inline __host__ __device__ int4 operator/(int3 x, int4 y){
    return make_int4(x.x/y.x, x.y/y.y, x.z/y.z, y.w);
}
inline __host__ __device__ int4 operator/(int4 x, int4 y){
    return make_int4(x.x/y.x, x.y/y.y, x.z/y.z);
}


/////////////////////////////////////
// |
/////////////////////////////////////


inline __host__ __device__ int2 operator|(int x, int2 y){
    return make_int2(x|y.x, x|y.y);
}
inline __host__ __device__ int2 operator|(int2 x, int y){
    return make_int2(x.x|y, x.y|y);
}
inline __host__ __device__ int2 operator|(int2 x, int2 y){
    return make_int2(x.x|y.x, x.y|y.y);
}
inline __host__ __device__ int3 operator|(int3 x, int y){
    return make_int3(x.x|y, x.y|y, x.z|y);
}
inline __host__ __device__ int3 operator|(int x, int3 y){
    return make_int3(x|y.x, x|y.y, x|y.z);
}
inline __host__ __device__ int3 operator|(int3 x, int2 y){
    return make_int3(x.x|y.x, x.y|y.y, x.z);
}
inline __host__ __device__ int3 operator|(int2 x, int3 y){
    return make_int3(x.x|y.x, x.y|y.y, y.z);
}
inline __host__ __device__ int3 operator|(int3 x, int3 y){
    return make_int3(x.x|y.x, x.y|y.y, x.z|y.z);
}
inline __host__ __device__ int4 operator|(int4 x, int y){
    return make_int4(x.x|y, x.y|y, x.z|y, x.w|y);
}
inline __host__ __device__ int4 operator|(int x, int4 y){
    return make_int4(x|y.x, x|y.y, x|y.z, x|y.w);
}
inline __host__ __device__ int4 operator|(int4 x, int2 y){
    return make_int4(x.x|y.x, x.y|y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator|(int2 x, int4 y){
    return make_int4(x.x|y.x, x.y|y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator|(int4 x, int3 y){
    return make_int4(x.x|y.x, x.y|y.y, x.z|y.z, x.w);
}
inline __host__ __device__ int4 operator|(int3 x, int4 y){
    return make_int4(x.x|y.x, x.y|y.y, x.z|y.z, y.w);
}
inline __host__ __device__ int4 operator|(int4 x, int4 y){
    return make_int4(x.x|y.x, x.y|y.y, x.z|y.z);
}


/////////////////////////////////////
// ^
/////////////////////////////////////


inline __host__ __device__ int2 operator^(int x, int2 y){
    return make_int2(x^y.x, x^y.y);
}
inline __host__ __device__ int2 operator^(int2 x, int y){
    return make_int2(x.x^y, x.y^y);
}
inline __host__ __device__ int2 operator^(int2 x, int2 y){
    return make_int2(x.x^y.x, x.y^y.y);
}
inline __host__ __device__ int3 operator^(int3 x, int y){
    return make_int3(x.x^y, x.y^y, x.z^y);
}
inline __host__ __device__ int3 operator^(int x, int3 y){
    return make_int3(x^y.x, x^y.y, x^y.z);
}
inline __host__ __device__ int3 operator^(int3 x, int2 y){
    return make_int3(x.x^y.x, x.y^y.y, x.z);
}
inline __host__ __device__ int3 operator^(int2 x, int3 y){
    return make_int3(x.x^y.x, x.y^y.y, y.z);
}
inline __host__ __device__ int3 operator^(int3 x, int3 y){
    return make_int3(x.x^y.x, x.y^y.y, x.z^y.z);
}
inline __host__ __device__ int4 operator^(int4 x, int y){
    return make_int4(x.x^y, x.y^y, x.z^y, x.w^y);
}
inline __host__ __device__ int4 operator^(int x, int4 y){
    return make_int4(x^y.x, x^y.y, x^y.z, x^y.w);
}
inline __host__ __device__ int4 operator^(int4 x, int2 y){
    return make_int4(x.x^y.x, x.y^y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator^(int2 x, int4 y){
    return make_int4(x.x^y.x, x.y^y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator^(int4 x, int3 y){
    return make_int4(x.x^y.x, x.y^y.y, x.z^y.z, x.w);
}
inline __host__ __device__ int4 operator^(int3 x, int4 y){
    return make_int4(x.x^y.x, x.y^y.y, x.z^y.z, y.w);
}
inline __host__ __device__ int4 operator^(int4 x, int4 y){
    return make_int4(x.x^y.x, x.y^y.y, x.z^y.z);
}


/////////////////////////////////////
// ~
/////////////////////////////////////


inline __host__ __device__ int2 operator~(int2 v){
    return make_int2(~v.x, ~v.y);
}
inline __host__ __device__ int3 operator~(int3 v){
    return make_int3(~v.x, ~v.y, ~v.z);
}
inline __host__ __device__ int4 operator~(int4 v){
    return make_int4(~v.x, ~v.y, ~v.z, ~v.w);
}


/////////////////////////////////////
// <<
/////////////////////////////////////


inline __host__ __device__ int2 operator<<(int x, int2 y){
    return make_int2(x<<y.x, x<<y.y);
}
inline __host__ __device__ int2 operator<<(int2 x, int y){
    return make_int2(x.x<<y, x.y<<y);
}
inline __host__ __device__ int2 operator<<(int2 x, int2 y){
    return make_int2(x.x<<y.x, x.y<<y.y);
}
inline __host__ __device__ int3 operator<<(int3 x, int y){
    return make_int3(x.x<<y, x.y<<y, x.z<<y);
}
inline __host__ __device__ int3 operator<<(int x, int3 y){
    return make_int3(x<<y.x, x<<y.y, x<<y.z);
}
inline __host__ __device__ int3 operator<<(int3 x, int2 y){
    return make_int3(x.x<<y.x, x.y<<y.y, x.z);
}
inline __host__ __device__ int3 operator<<(int2 x, int3 y){
    return make_int3(x.x<<y.x, x.y<<y.y, y.z);
}
inline __host__ __device__ int3 operator<<(int3 x, int3 y){
    return make_int3(x.x<<y.x, x.y<<y.y, x.z<<y.z);
}
inline __host__ __device__ int4 operator<<(int4 x, int y){
    return make_int4(x.x<<y, x.y<<y, x.z<<y, x.w<<y);
}
inline __host__ __device__ int4 operator<<(int x, int4 y){
    return make_int4(x<<y.x, x<<y.y, x<<y.z, x<<y.w);
}
inline __host__ __device__ int4 operator<<(int4 x, int2 y){
    return make_int4(x.x<<y.x, x.y<<y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator<<(int2 x, int4 y){
    return make_int4(x.x<<y.x, x.y<<y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator<<(int4 x, int3 y){
    return make_int4(x.x<<y.x, x.y<<y.y, x.z<<y.z, x.w);
}
inline __host__ __device__ int4 operator<<(int3 x, int4 y){
    return make_int4(x.x<<y.x, x.y<<y.y, x.z<<y.z, y.w);
}
inline __host__ __device__ int4 operator<<(int4 x, int4 y){
    return make_int4(x.x<<y.x, x.y<<y.y, x.z<<y.z);
}


/////////////////////////////////////
// >>
/////////////////////////////////////


inline __host__ __device__ int2 operator>>(int x, int2 y){
    return make_int2(x>>y.x, x>>y.y);
}
inline __host__ __device__ int2 operator>>(int2 x, int y){
    return make_int2(x.x>>y, x.y>>y);
}
inline __host__ __device__ int2 operator>>(int2 x, int2 y){
    return make_int2(x.x>>y.x, x.y>>y.y);
}
inline __host__ __device__ int3 operator>>(int3 x, int y){
    return make_int3(x.x>>y, x.y>>y, x.z>>y);
}
inline __host__ __device__ int3 operator>>(int x, int3 y){
    return make_int3(x>>y.x, x>>y.y, x>>y.z);
}
inline __host__ __device__ int3 operator>>(int3 x, int2 y){
    return make_int3(x.x>>y.x, x.y>>y.y, x.z);
}
inline __host__ __device__ int3 operator>>(int2 x, int3 y){
    return make_int3(x.x>>y.x, x.y>>y.y, y.z);
}
inline __host__ __device__ int3 operator>>(int3 x, int3 y){
    return make_int3(x.x>>y.x, x.y>>y.y, x.z>>y.z);
}
inline __host__ __device__ int4 operator>>(int4 x, int y){
    return make_int4(x.x>>y, x.y>>y, x.z>>y, x.w>>y);
}
inline __host__ __device__ int4 operator>>(int x, int4 y){
    return make_int4(x>>y.x, x>>y.y, x>>y.z, x>>y.w);
}
inline __host__ __device__ int4 operator>>(int4 x, int2 y){
    return make_int4(x.x>>y.x, x.y>>y.y, x.z, x.w);
}
inline __host__ __device__ int4 operator>>(int2 x, int4 y){
    return make_int4(x.x>>y.x, x.y>>y.y, y.z, y.w);
}
inline __host__ __device__ int4 operator>>(int4 x, int3 y){
    return make_int4(x.x>>y.x, x.y>>y.y, x.z>>y.z, x.w);
}
inline __host__ __device__ int4 operator>>(int3 x, int4 y){
    return make_int4(x.x>>y.x, x.y>>y.y, x.z>>y.z, y.w);
}
inline __host__ __device__ int4 operator>>(int4 x, int4 y){
    return make_int4(x.x>>y.x, x.y>>y.y, x.z>>y.z);
}



/////////////////////////////////////
// min
/////////////////////////////////////


inline  __host__ __device__ int2 min(int x, int2 y){
    return make_int2(min(x,y.x), min(x,y.y));
}
inline  __host__ __device__ int2 min(int2 x, int y){
    return make_int2(min(x.x,y), min(x.y,y));
}
inline  __host__ __device__ int2 min(int2 x, int2 y){
    return make_int2(min(x.x,y.x), min(x.y,y.y));
}
inline __host__ __device__ int3 min(int x, int3 y){
    return make_int3(min(x,y.x), min(x,y.y), min(x,y.z));
}
inline __host__ __device__ int3 min(int3 x, int y){
    return make_int3(min(x.x,y), min(x.y,y), min(x.z,y));
}
inline __host__ __device__ int3 min(int3 x, int3 y){
    return make_int3(min(x.x,y.x), min(x.y,y.y), min(x.z,y.z));
}
inline  __host__ __device__ int4 min(int x, int4 y){
    return make_int4(min(x,y.x), min(x,y.y), min(x,y.z), min(x,y.w));
}
inline  __host__ __device__ int4 min(int4 x, int y){
    return make_int4(min(x.x,y), min(x.y,y), min(x.z,y), min(x.w,y));
}
inline  __host__ __device__ int4 min(int4 x, int4 y){
    return make_int4(min(x.x,y.x), min(x.y,y.y), min(x.z,y.z), min(x.w,y.w));
}


/////////////////////////////////////
// max
/////////////////////////////////////


inline  __host__ __device__ int2 max(int x, int2 y){
    return make_int2(max(x,y.x), max(x,y.y));
}
inline  __host__ __device__ int2 max(int2 x, int y){
    return make_int2(max(x.x,y), max(x.y,y));
}
inline  __host__ __device__ int2 max(int2 x, int2 y){
    return make_int2(max(x.x,y.x), max(x.y,y.y));
}
inline __host__ __device__ int3 max(int x, int3 y){
    return make_int3(max(x,y.x), max(x,y.y), max(x,y.z));
}
inline __host__ __device__ int3 max(int3 x, int y){
    return make_int3(max(x.x,y), max(x.y,y), max(x.z,y));
}
inline __host__ __device__ int3 max(int3 x, int3 y){
    return make_int3(max(x.x,y.x), max(x.y,y.y), max(x.z,y.z));
}
inline  __host__ __device__ int4 max(int x, int4 y){
    return make_int4(max(x,y.x), max(x,y.y), max(x,y.z), max(x,y.w));
}
inline  __host__ __device__ int4 max(int4 x, int y){
    return make_int4(max(x.x,y), max(x.y,y), max(x.z,y), max(x.w,y));
}
inline  __host__ __device__ int4 max(int4 x, int4 y){
    return make_int4(max(x.x,y.x), max(x.y,y.y), max(x.z,y.z), max(x.w,y.w));
}


/////////////////////////////////////
// dot
/////////////////////////////////////


inline __host__ __device__ int dot(int2 x, int2 y){
    return x.x*y.x + y.y*y.y;
}
inline __host__ __device__ int dot(int3 x, int3 y){
    return x.x*y.x + x.y*y.y + x.z*y.z;
}
inline __host__ __device__ int dot(int4 x, int4 y){
    return x.x*y.x + x.y*y.y + x.z*y.z + x.w*y.w;
}


/////////////////////////////////////
// abs
/////////////////////////////////////


inline __host__ __device__ int2 abs(int2 v){
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v){
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v){
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}


/////////////////////////////////////
// sum
/////////////////////////////////////


inline __host__ __device__ int sum(int2 x){
    return x.x + x.y;
}
inline __host__ __device__ int sum(int3 x){
    return x.x + x.y + x.z;
}
inline __host__ __device__ int sum(int4 x){
    return x.x + x.y + x.z + x.w;
}


#endif


