//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_SPHERE_H
#define RAYTRACER_SPHERE_H

#include "Hittable.h"
#include "Vector3.h"

class Sphere : public Hittable {

    public:
        Sphere() { }
        Sphere(point3 cen, double r, std::shared_ptr<Material> m) : center(cen), radius(r), mat_ptr(m) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

    public:
        point3 center;
        double radius;
        std::shared_ptr<Material> mat_ptr;

};

bool Sphere::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {

    Vector3 oc = r.origin() - center;

    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;

    if(discriminant > 0){

        auto root = std::sqrt(discriminant);

        auto temp = (-half_b - root)/a;

        if(temp < t_max && temp > t_min){

            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r,outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }

        temp = (-half_b + root)/a;

        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r,outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }

    }

    return false;

}


#endif //RAYTRACER_SPHERE_H
