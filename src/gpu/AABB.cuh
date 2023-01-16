#ifndef RAYTRACER_AABB_cuh
#define RAYTRACER_AABB_cuh
 
#include "Vector3.cuh"
#include "Ray.cuh"

class AABB {
    public:
        __device__ AABB() {}
        __device__ AABB(const Point3& a, const Point3& b) {
            minimum = a;
            maximum = b;
        }

        __device__ bool hit(const Ray& r, float tMin, float tMax) const {
            for(int a = 0; a < 3; a++) {
                auto invDir = 1.0f/r.direction()[a];

                auto t0 = (min()[a] - r.origin()[a]) * invDir;
                auto t1 = (max()[a] - r.origin()[a]) * invDir;

                if(invDir < 0.0f) std::swap(t0, t1);

                tMin = t0 > tMin ? t0 : tMin;
                tMax = t1 < tMax ? t1 : tMax;

                if(tMax <= tMin) return false;

            }

            return true;
        }

        __device__ Point3 min() const { return minimum; }
        __device__ Point3 max() const { return maximum; }

        Point3 minimum;
        Point3 maximum;
};

__device__ AABB sorroundingBox(AABB box0, AABB box1) {
    Point3 small(std::fmin(box0.min().x(), box1.min().x()),
                 std::fmin(box0.min().y(), box1.min().y()),
                 std::fmin(box0.min().z(), box1.min().z()));
    Point3 big(std::fmax(box0.max().x(), box1.max().x()),
               std::fmax(box0.max().y(), box1.max().y()),
               std::fmax(box0.max().z(), box1.max().z()));

    return AABB(small,big);
}
 
#endif // RAYTRACER_AABB_cuh