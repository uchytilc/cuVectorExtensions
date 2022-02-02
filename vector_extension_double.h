#ifndef VECTOR_EXTENSION_DOUBLE_H
#define VECTOR_EXTENSION_DOUBLE_H


#include "vector_extension_utils.h"


/////////////////////////////////////
// make_vector
/////////////////////////////////////


inline __host__ __device__ double2 make_double2(double x){
    return make_double2(x, x);
}
inline __host__ __device__ double2 make_double2(double2 v){
    return make_double2(v.x, v.y);
}
inline __host__ __device__ double2 make_double2(double3 v){
    return make_double2(v.x, v.y);
}
inline __host__ __device__ double2 make_double2(double4 v){
    return make_double2(v.x, v.y);
}

inline __host__ __device__ double3 make_double3(double x){
    return make_double3(x, x, x);
}
inline __host__ __device__ double3 make_double3(double x, double y){
    return make_double3(x, y, 0.0);
}
inline __host__ __device__ double3 make_double3(double2 v){
    return make_double3(v.x, v.y, 0.0);
}
inline __host__ __device__ double3 make_double3(double2 v, double x){
    return make_double3(v.x, v.y, x);
}
inline __host__ __device__ double3 make_double3(double x, double2 v){
    return make_double3(x, v.x, v.y);
}
inline __host__ __device__ double3 make_double3(double3 v){
    return make_double3(v.x, v.y, v.z);
}
inline __host__ __device__ double3 make_double3(double4 v){
    return make_double3(v.x, v.y, v.z);
}

inline __host__ __device__ double4 make_double4(double x){
    return make_double4(x, x, x, x);
}
inline __host__ __device__ double4 make_double4(double x, double y){
    return make_double4(x, y, 0.0, 0.0);
}
inline __host__ __device__ double4 make_double4(double2 v){
    return make_double4(v.x, v.y, 0.0, 0.0);
}
inline __host__ __device__ double4 make_double4(double x, double y, double z){
    return make_double4(x, y, z, 0.0);
}
inline __host__ __device__ double4 make_double4(double2 x, double y){
    return make_double4(x.x, x.y, y, 0.0);
}
inline __host__ __device__ double4 make_double4(double x, double2 y){
    return make_double4(x, y.x, y.y, 0.0);
}
inline __host__ __device__ double4 make_double4(double3 v){
    return make_double4(v.x, v.y, v.z, 0.0);
}
inline __host__ __device__ double4 make_double4(double x, double y, double2 z){
    return make_double4(x, y, z.x, z.y);
}
inline __host__ __device__ double4 make_double4(double x, double2 y, double z){
    return make_double4(x, y.x, y.y, z);
}
inline __host__ __device__ double4 make_double4(double2 x, double y, double z){
    return make_double4(x.x, x.y, y, z);
}
inline __host__ __device__ double4 make_double4(double2 x, double2 y){
    return make_double4(x.x, x.y, y.x, y.y);
}
inline __host__ __device__ double4 make_double4(double3 v, double w){
    return make_double4(v.x, v.y, v.z, w);
}
inline __host__ __device__ double4 make_double4(double x, double3 v){
    return make_double4(x, v.x, v.y, v.z);
}
inline __host__ __device__ double4 make_double4(double4 v){
    return make_double4(v.x, v.y, v.z, v.w);
}


/////////////////////////////////////
// negate
/////////////////////////////////////


inline __host__ __device__ double2 operator-(double2 v){ //const double2& v
    return make_double2(-v.x, -v.y);
}

inline __host__ __device__ double3 operator-(double3 v){
    return make_double3(-v.x, -v.y, -v.z);
}

inline __host__ __device__ double4 operator-(double4 v){
    return make_double4(-v.x, -v.y, -v.z, -v.w);
}





// inline __host__ __device__ double2 operator*(double x, double2 y){
//     return make_double2(x*y.x, x*y.y);
// }
// inline __host__ __device__ double2 operator*(double2 x, double y){
//     return make_double2(x.x*y, x.y*y);
// }
inline __host__ __device__ double2 operator*(double2 x, double2 y){
    return make_double2(x.x*y.x, x.y*y.y);
}
// inline __host__ __device__ double3 operator*(double3 x, double y){
//     return make_double3(x.x*y, x.y*y, x.z*y);
// }
inline __host__ __device__ double3 operator*(double x, double3 y){
    return make_double3(x*y.x, x*y.y, x*y.z);
}
// inline __host__ __device__ double3 operator*(double3 x, double2 y){
//     return make_double3(x.x*y.x, x.y*y.y, x.z);
// }
// inline __host__ __device__ double3 operator*(double2 x, double3 y){
//     return make_double3(x.x*y.x, x.y*y.y, y.z);
// }
inline __host__ __device__ double3 operator*(double3 x, double3 y){
    return make_double3(x.x*y.x, x.y*y.y, x.z*y.z);
}
inline __host__ __device__ double4 operator*(double4 x, double y){
    return make_double4(x.x*y, x.y*y, x.z*y, x.w*y);
}
inline __host__ __device__ double4 operator*(double x, double4 y){
    return make_double4(x*y.x, x*y.y, x*y.z, x*y.w);
}
// inline __host__ __device__ double4 operator*(double4 x, double2 y){
//     return make_double4(x.x*y.x, x.y*y.y, x.z, x.w);
// }
// inline __host__ __device__ double4 operator*(double2 x, double4 y){
//     return make_double4(x.x*y.x, x.y*y.y, y.z, y.w);
// }
// inline __host__ __device__ double4 operator*(double4 x, double3 y){
//     return make_double4(x.x*y.x, x.y*y.y, x.z*y.z, x.w);
// }
// inline __host__ __device__ double4 operator*(double3 x, double4 y){
//     return make_double4(x.x*y.x, x.y*y.y, x.z*y.z, y.w);
// }
inline __host__ __device__ double4 operator*(double4 x, double4 y){
    return make_double4(x.x*y.x, x.y*y.y, x.z*y.z);
}



// inline __host__ __device__ double2 operator/(double x, double2 y){
//     return make_double2(x/y.x, x/y.y);
// }
// inline __host__ __device__ double2 operator/(double2 x, double y){
//     return make_double2(x.x/y, x.y/y);
// }
// inline __host__ __device__ double2 operator/(double2 x, double2 y){
//     return make_double2(x.x/y.x, x.y/y.y);
// }
inline __host__ __device__ double3 operator/(double3 x, double y){
    return make_double3(x.x/y, x.y/y, x.z/y);
}
// inline __host__ __device__ double3 operator/(double x, double3 y){
//     return make_double3(x/y.x, x/y.y, x/y.z);
// }
// inline __host__ __device__ double3 operator/(double3 x, double2 y){
//     return make_double3(x.x/y.x, x.y/y.y, x.z);
// }
// inline __host__ __device__ double3 operator/(double2 x, double3 y){
//     return make_double3(x.x/y.x, x.y/y.y, y.z);
// }
inline __host__ __device__ double3 operator/(double3 x, double3 y){
    return make_double3(x.x/y.x, x.y/y.y, x.z/y.z);
}
// inline __host__ __device__ double4 operator/(double4 x, double y){
//     return make_double4(x.x/y, x.y/y, x.z/y, x.w/y);
// }
// inline __host__ __device__ double4 operator/(double x, double4 y){
//     return make_double4(x/y.x, x/y.y, x/y.z, x/y.w);
// }
// inline __host__ __device__ double4 operator/(double4 x, double2 y){
//     return make_double4(x.x/y.x, x.y/y.y, x.z, x.w);
// }
// inline __host__ __device__ double4 operator/(double2 x, double4 y){
//     return make_double4(x.x/y.x, x.y/y.y, y.z, y.w);
// }
// inline __host__ __device__ double4 operator/(double4 x, double3 y){
//     return make_double4(x.x/y.x, x.y/y.y, x.z/y.z, x.w);
// }
// inline __host__ __device__ double4 operator/(double3 x, double4 y){
//     return make_double4(x.x/y.x, x.y/y.y, x.z/y.z, y.w);
// }
inline __host__ __device__ double4 operator/(double4 x, double4 y){
    return make_double4(x.x/y.x, x.y/y.y, x.z/y.z);
}



// inline __host__ __device__ double2 operator+(double x, double2 y){
//     return make_double2(x + y.x, x + y.y);
// }
// inline __host__ __device__ double2 operator+(double2 x, double y){
//     return make_double2(x.x + y, x.y + y);
// }
inline __host__ __device__ double2 operator+(double2 x, double2 y){
    return make_double2(x.x + y.x, x.y + y.y);
}
inline __host__ __device__ double3 operator+(double3 x, double y){
    return make_double3(x.x + y, x.y + y, x.z + y);
}
inline __host__ __device__ double3 operator+(double x, double3 y){
    return make_double3(x + y.x, x + y.y, x + y.z);
}
// inline __host__ __device__ double3 operator+(double3 x, double2 y){
//     return make_double3(x.x + y.x, x.y + y.y, x.z);
// }
// inline __host__ __device__ double3 operator+(double2 x, double3 y){
//     return make_double3(x.x + y.x, x.y + y.y, y.z);
// }
inline __host__ __device__ double3 operator+(double3 x, double3 y){
    return make_double3(x.x + y.x, x.y + y.y, x.z + y.z);
}
// inline __host__ __device__ double4 operator+(double4 x, double y){
//     return make_double4(x.x + y, x.y + y, x.z + y, x.w + y);
// }
// inline __host__ __device__ double4 operator+(double x, double4 y){
//     return make_double4(x + y.x, x + y.y, x + y.z, x + y.w);
// }
// inline __host__ __device__ double4 operator+(double4 x, double2 y){
//     return make_double4(x.x + y.x, x.y + y.y, x.z, x.w);
// }
// inline __host__ __device__ double4 operator+(double2 x, double4 y){
//     return make_double4(x.x + y.x, x.y + y.y, y.z, y.w);
// }
// inline __host__ __device__ double4 operator+(double4 x, double3 y){
//     return make_double4(x.x + y.x, x.y + y.y, x.z + y.z, x.w);
// }
// inline __host__ __device__ double4 operator+(double3 x, double4 y){
//     return make_double4(x.x + y.x, x.y + y.y, x.z + y.z, y.w);
// }
inline __host__ __device__ double4 operator+(double4 x, double4 y){
    return make_double4(x.x + y.x, x.y + y.y, x.z + y.z);
}


// inline __host__ __device__ double2 operator-(double x, double2 y){
//     return make_double2(x - y.x, x - y.y);
// }
// inline __host__ __device__ double2 operator-(double2 x, double y){
//     return make_double2(x.x - y, x.y - y);
// }
inline __host__ __device__ double2 operator-(double2 x, double2 y){
    return make_double2(x.x - y.x, x.y - y.y);
}
inline __host__ __device__ double3 operator-(double3 x, double y){
    return make_double3(x.x - y, x.y - y, x.z - y);
}
// inline __host__ __device__ double3 operator-(double x, double3 y){
//     return make_double3(x - y.x, x - y.y, x - y.z);
// }
// inline __host__ __device__ double3 operator-(double3 x, double2 y){
//     return make_double3(x.x - y.x, x.y - y.y, x.z);
// }
// inline __host__ __device__ double3 operator-(double2 x, double3 y){
//     return make_double3(x.x - y.x, x.y - y.y, y.z);
// }
inline __host__ __device__ double3 operator-(double3 x, double3 y){
    return make_double3(x.x - y.x, x.y - y.y, x.z - y.z);
}
// inline __host__ __device__ double4 operator-(double4 x, double y){
//     return make_double4(x.x - y, x.y - y, x.z - y, x.w - y);
// }
// inline __host__ __device__ double4 operator-(double x, double4 y){
//     return make_double4(x - y.x, x - y.y, x - y.z, x - y.w);
// }
// inline __host__ __device__ double4 operator-(double4 x, double2 y){
//     return make_double4(x.x - y.x, x.y - y.y, x.z, x.w);
// }
// inline __host__ __device__ double4 operator-(double2 x, double4 y){
//     return make_double4(x.x - y.x, x.y - y.y, y.z, y.w);
// }
// inline __host__ __device__ double4 operator-(double4 x, double3 y){
//     return make_double4(x.x - y.x, x.y - y.y, x.z - y.z, x.w);
// }
// inline __host__ __device__ double4 operator-(double3 x, double4 y){
//     return make_double4(x.x - y.x, x.y - y.y, x.z - y.z, y.w);
// }
// inline __host__ __device__ double4 operator-(double4 x, double4 y){
//     return make_double4(x.x - y.x, x.y - y.y, x.z - y.z);
// }


inline __host__ __device__ uint3 operator>(double3 a, double b) {
    return make_uint3(a.x > b, a.y > b, a.z > b);
}

inline __host__ __device__ uint3 operator<(double3 a, double b) {
    return make_uint3(a.x < b, a.y < b, a.z < b);
}

inline __host__ __device__ uint3 operator>=(double3 a, double b) {
    return make_uint3(a.x >= b, a.y >= b, a.z >= b);
}

inline __host__ __device__ uint3 operator<=(double3 a, double b) {
    return make_uint3(a.x <= b, a.y <= b, a.z <= b);
}



inline __host__ __device__ uint3 operator>(double3 a, double3 b) {
    return make_uint3(a.x > b.x, a.y > b.y, a.z > b.z);
}

inline __host__ __device__ uint3 operator<(double3 a, double3 b) {
    return make_uint3(a.x < b.x, a.y < b.y, a.z < b.z);
}

inline __host__ __device__ uint3 operator>=(double3 a, double3 b) {
    return make_uint3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

inline __host__ __device__ uint3 operator<=(double3 a, double3 b) {
    return make_uint3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}



/////////////////////////////////////
// min
/////////////////////////////////////


inline  __host__ __device__ double2 min(double x, double2 y){
    return make_double2(min(x,y.x), min(x,y.y));
}
inline  __host__ __device__ double2 min(double2 x, double y){
    return make_double2(min(x.x,y), min(x.y,y));
}
inline  __host__ __device__ double2 min(double2 x, double2 y){
    return make_double2(min(x.x,y.x), min(x.y,y.y));
}
inline __host__ __device__ double3 min(double x, double3 y){
    return make_double3(min(x,y.x), min(x,y.y), min(x,y.z));
}
inline __host__ __device__ double3 min(double3 x, double y){
    return make_double3(min(x.x,y), min(x.y,y), min(x.z,y));
}
inline __host__ __device__ double3 min(double3 x, double3 y){
    return make_double3(min(x.x,y.x), min(x.y,y.y), min(x.z,y.z));
}
inline  __host__ __device__ double4 min(double x, double4 y){
    return make_double4(min(x,y.x), min(x,y.y), min(x,y.z), min(x,y.w));
}
inline  __host__ __device__ double4 min(double4 x, double y){
    return make_double4(min(x.x,y), min(x.y,y), min(x.z,y), min(x.w,y));
}
inline  __host__ __device__ double4 min(double4 x, double4 y){
    return make_double4(min(x.x,y.x), min(x.y,y.y), min(x.z,y.z), min(x.w,y.w));
}

inline  __host__ __device__ double min(double2 v){
    return min(v.x, v.y);
}
inline  __host__ __device__ double min(double3 v){
    return min(v.x, min(v.y, v.z));
}
inline  __host__ __device__ double min(double4 v){
    return min(v.x, min(v.y, min(v.z, v.w)));
}

/////////////////////////////////////
// max
/////////////////////////////////////


inline  __host__ __device__ double2 max(double x, double2 y){
    return make_double2(max(x,y.x), max(x,y.y));
}
inline  __host__ __device__ double2 max(double2 x, double y){
    return make_double2(max(x.x,y), max(x.y,y));
}
inline  __host__ __device__ double2 max(double2 x, double2 y){
    return make_double2(max(x.x,y.x), max(x.y,y.y));
}
inline __host__ __device__ double3 max(double x, double3 y){
    return make_double3(max(x,y.x), max(x,y.y), max(x,y.z));
}
inline __host__ __device__ double3 max(double3 x, double y){
    return make_double3(max(x.x,y), max(x.y,y), max(x.z,y));
}
inline __host__ __device__ double3 max(double3 x, double3 y){
    return make_double3(max(x.x,y.x), max(x.y,y.y), max(x.z,y.z));
}
inline  __host__ __device__ double4 max(double x, double4 y){
    return make_double4(max(x,y.x), max(x,y.y), max(x,y.z), max(x,y.w));
}
inline  __host__ __device__ double4 max(double4 x, double y){
    return make_double4(max(x.x,y), max(x.y,y), max(x.z,y), max(x.w,y));
}
inline  __host__ __device__ double4 max(double4 x, double4 y){
    return make_double4(max(x.x,y.x), max(x.y,y.y), max(x.z,y.z), max(x.w,y.w));
}

inline  __host__ __device__ double max(double2 v){
    return max(v.x, v.y);
}
inline  __host__ __device__ double max(double3 v){
    return max(v.x, max(v.y, v.z));
}
inline  __host__ __device__ double max(double4 v){
    return max(v.x, max(v.y, max(v.z, v.w)));
}



/////////////////////////////////////
// floor
/////////////////////////////////////


inline __host__ __device__ double2 floor(double2 v){
    return make_double2(floor(v.x), floor(v.y));
}
inline __host__ __device__ double3 floor(double3 v){
    return make_double3(floor(v.x), floor(v.y), floor(v.z));
}
inline __host__ __device__ double4 floor(double4 v){
    return make_double4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}


/////////////////////////////////////
// clamp
/////////////////////////////////////


inline __device__ __host__ double clamp(double f, double lo, double hi){
    return fmaxf(lo, fminf(f, hi));
}
inline __device__ __host__ double2 clamp(double2 v, double lo, double hi){
    return make_double2(clamp(v.x, lo, hi), clamp(v.y, lo, hi));
}
inline __device__ __host__ double2 clamp(double2 v, double2 lo, double2 hi){
    return make_double2(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y));
}
inline __device__ __host__ double3 clamp(double3 v, double lo, double hi){
    return make_double3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}
inline __device__ __host__ double3 clamp(double3 v, double lo, double3 hi){
    return make_double3(clamp(v.x, lo, hi.x), clamp(v.y, lo, hi.y), clamp(v.z, lo, hi.z));
}
inline __device__ __host__ double3 clamp(double3 v, double3 lo, double hi){
    return make_double3(clamp(v.x, lo.x, hi), clamp(v.y, lo.y, hi), clamp(v.z, lo.z, hi));
}
inline __device__ __host__ double3 clamp(double3 v, double3 lo, double3 hi){
    return make_double3(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z));
}
inline __device__ __host__ double4 clamp(double4 v, double lo, double hi){
    return make_double4(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi), clamp(v.w, lo, hi));
}
inline __device__ __host__ double4 clamp(double4 v, double4 lo, double4 hi){
    return make_double4(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z), clamp(v.w, lo.w, hi.w));
}


/////////////////////////////////////
// abs
/////////////////////////////////////


inline __host__ __device__ double2 abs(double2 v){
    return make_double2(abs(v.x), abs(v.y));
}
inline __host__ __device__ double3 abs(double3 v){
    return make_double3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ double4 abs(double4 v){
    return make_double4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}


/////////////////////////////////////
// sum
/////////////////////////////////////


inline __host__ __device__ double sum(double2 x){
    return x.x + x.y;
}
inline __host__ __device__ double sum(double3 x){
    return x.x + x.y + x.z;
}
inline __host__ __device__ double sum(double4 x){
    return x.x + x.y + x.z + x.w;
}


/////////////////////////////////////
// dot
/////////////////////////////////////


inline __host__ __device__ double dot(double2 x, double2 y){
    return sum(x*y);
}
inline __host__ __device__ double dot(double3 x, double3 y){
    return sum(x*y);
}
inline __host__ __device__ double dot(double4 x, double4 y){
    return sum(x*y);
}


/////////////////////////////////////
// l2-norm
/////////////////////////////////////


inline __host__ __device__ double length(double2 v){
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double3 v){
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double4 v){
    return sqrt(dot(v, v));
}


/////////////////////////////////////
// normalize
/////////////////////////////////////


inline __host__ __device__ double3 normalize(double3 v){
    return v/length(v);
}


/////////////////////////////////////
// fmod
/////////////////////////////////////

inline __host__ __device__ double2 fmod(double2 a, double2 b){
    return make_double2(fmod(a.x, b.x), fmod(a.y, b.y));
}

inline __host__ __device__ double2 fmod(double2 a, double b){
    return make_double2(fmod(a.x, b), fmod(a.y, b));
}



#endif