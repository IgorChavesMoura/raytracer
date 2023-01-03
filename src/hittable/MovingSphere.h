#ifndef RAYTRACER_MOVING_SPHERE_H
#define RAYTRACER_MOVING_SPHERE_H

#include "Hittable.h"
#include "../Vector3.h"

class MovingSphere : public Hittable {
    public:
        MovingSphere() {}
        MovingSphere(Point3 c0, Point3 c1, double t0, double t1, double r, std::shared_ptr<Material> m) : center0(c0), center1(c1), time0(t0), time1(t1), radius(r), mat_ptr(m) {}

        virtual bool hit(
                const Ray& r, double tMin, double tMax, HitRecord& rec) const override;
        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override;

        Point3 center(double time) const;

    public:
        Point3 center0, center1;
        double time0, time1;
        double radius;
        std::shared_ptr<Material> mat_ptr;
};

Point3 MovingSphere::center(double time) const {
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

bool MovingSphere::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {
    Vector3 oc = r.origin() - center(r.time());

    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;

    if(discriminant < 0) return false;

    auto sqrtDiscriminant = sqrt(discriminant);

    auto root = (-half_b - sqrtDiscriminant) / a;
    if(root < tMin || root > tMax) {
        root = (-half_b + sqrtDiscriminant) / a;
        if(root < tMin || root > tMax) {
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    auto outward_normal = (rec.p - center(r.time())) / radius;
    rec.setFaceNormal(r, outward_normal);
    rec.matPtr = mat_ptr;

    return true;
}

bool MovingSphere::boundingBox(double t0, double t1, AABB &outputBox) const {
    AABB box0(center(t0) - Vector3(radius,radius,radius),
              center(t0) + Vector3(radius,radius,radius));

    AABB box1(center(t1) - Vector3(radius,radius,radius),
              center(t1) + Vector3(radius,radius,radius));

    outputBox = sorroundingBox(box0, box1);

    return true;
}

#endif //RAYTRACER_MOVING_SPHERE_H