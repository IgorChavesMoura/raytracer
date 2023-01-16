#ifndef RAYTRACER_Ray_cuh
#define RAYTRACER_Ray_cuh
 
#include "Vector3.cuh"

class Ray {

    public:

        __device__ Ray() { }
        __device__ Ray(const Point3& origin, const Vector3& direction, float time = 0.0f) : orig(origin), dir(direction), tm(time) { }

        __device__ Point3 origin() const {

            return orig;

        }

        __device__ Vector3 direction() const {

            return dir;

        }

        __device__ float time() const {
            
            return tm;

        }

        __device__ Point3 at(float t) const {

            return orig + t*dir;

        }


    public:
        Point3 orig;
        Vector3 dir;
        float tm;

};

#endif // RAYTRACER_Ray_cuh