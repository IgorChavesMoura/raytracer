//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_HITTABLE_H
#define RAYTRACER_HITTABLE_H

#include "Ray.h"

class Material;

struct hit_record {

    point3 p;
    Vector3 normal;
    std::shared_ptr<Material> mat_ptr;
    double t;
    bool front_face;

    inline void set_face_normal(const Ray& r, const Vector3& outward_normal){

        front_face = dot(r.direction(), outward_normal) < 0;

        normal = front_face ? outward_normal : -outward_normal;

    }

};

class Hittable {

    public:
        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};


#endif //RAYTRACER_HITTABLE_H
