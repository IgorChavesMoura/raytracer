//
// Created by moura on 22/12/2022.
//

#ifndef RAYTRACER_AABB_H
#define RAYTRACER_AABB_H

#include "Vector3.h"
#include "Ray.h"

class AABB {
    public:
        AABB() {}
        AABB(const Point3& a, const Point3& b) {
            minimum = a;
            maximum = b;
        }

        bool hit(const Ray& r, double tMin, double tMax) const {
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

        Point3 min() const { return minimum; }
        Point3 max() const { return maximum; }

        Point3 minimum;
        Point3 maximum;
};

AABB sorroundingBox(AABB box0, AABB box1) {
    Point3 small(std::fmin(box0.min().x(), box1.min().x()),
                 std::fmin(box0.min().y(), box1.min().y()),
                 std::fmin(box0.min().z(), box1.min().z()));
    Point3 big(std::fmax(box0.max().x(), box1.max().x()),
               std::fmax(box0.max().y(), box1.max().y()),
               std::fmax(box0.max().z(), box1.max().z()));

    return AABB(small,big);
}

#endif //RAYTRACER_AABB_H
