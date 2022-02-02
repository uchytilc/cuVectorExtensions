#ifndef VECTOR_EXTENSION_ULONGLONG_H
#define VECTOR_EXTENSION_ULONGLONG_H


////////////////////////////////////////////////////////////////////////////////
// sum
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ ulonglong sum(ulonglong2 x){
    return x.x + x.y;
}
inline __host__ __device__ ulonglong sum(ulonglong3 x){
    return x.x + x.y + x.z;
}
inline __host__ __device__ ulonglong sum(ulonglong4 x){
    return x.x + x.y + x.z + x.w;
}

////////////////////////////////////////////////////////////////////////////////
// product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ ulonglong prod(ulonglong2 x){
    return x.x * x.y;
}
inline __host__ __device__ ulonglong prod(ulonglong3 x){
    return x.x * x.y * x.z;
}
inline __host__ __device__ ulonglong prod(ulonglong4 x){
    return x.x * x.y * x.z * x.w;
}




#endif