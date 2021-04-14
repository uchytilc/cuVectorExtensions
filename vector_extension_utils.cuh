#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H

typedef unsigned int uint;

#define _CONCAT(a, b) a ## b
#define CONCAT(a, b) _CONCAT(a, b)

#define _VEC2(type) CONCAT(type, 2)
#define VEC2(type) _VEC2(type)

#define _VEC3(type) CONCAT(type, 3)
#define VEC3(type) _VEC3(type)

#define _VEC4(type) CONCAT(type, 4)
#define VEC4(type) _VEC4(type)

#define _MAKEVEC2(type) CONCAT(make_, VEC2(type))
#define MAKEVEC2(type) _MAKEVEC2(type)

#define _MAKEVEC3(type) CONCAT(make_, VEC3(type))
#define MAKEVEC3(type) _MAKEVEC3(type)

#define _MAKEVEC4(type) CONCAT(make_, VEC4(type))
#define MAKEVEC4(type) _MAKEVEC4(type)

#endif
