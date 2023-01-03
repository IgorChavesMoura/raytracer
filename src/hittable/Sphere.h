//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_SPHERE_H
#define RAYTRACER_SPHERE_H

#include "Hittable.h"
#include "../Vector3.h"

class Sphere : public Hittable {

    public:
        Sphere() { }
        Sphere(Point3 cen, double r, std::shared_ptr<Material> m) : center(cen), radius(r), matPtr(m) {};

        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;
        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override;

    public:
        Point3 center;
        double radius;
        std::shared_ptr<Material> matPtr;

    private:
        static void getSphereUv(const Point3& p, double& u, double& v) {
            auto theta = acos(-p.y());
            auto phi = atan2(-p.z(), p.x()) + pi;

            u = phi/(2*pi);
            v = theta/pi;
        }
};

bool Sphere::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {

    Vector3 oc = r.origin() - center;

    auto a = r.direction().length_squared();
    auto halfB = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = halfB * halfB - a * c;

    if(discriminant > 0){

        auto root = std::sqrt(discriminant);

        auto temp = (-halfB - root) / a;

        if(temp < tMax && temp > tMin){

            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(r, outward_normal);
            getSphereUv(outward_normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            return true;
        }

        temp = (-halfB + root) / a;

        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(r, outward_normal);
            getSphereUv(outward_normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            return true;
        }

    }

    return false;

}

bool Sphere::boundingBox(double t0, double t1, AABB &outputBox) const {
    outputBox = AABB(center - Vector3(radius,radius,radius), center + Vector3(radius,radius,radius));

    return true;
}


#endif //RAYTRACER_SPHERE_H
