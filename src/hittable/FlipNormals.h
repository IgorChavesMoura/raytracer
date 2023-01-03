//
// Created by moura on 27/12/2022.
//

#ifndef RAYTRACER_FLIPNORMALS_H
#define RAYTRACER_FLIPNORMALS_H

#include "Hittable.h"

class FlipNormals : public Hittable {
    public:
        FlipNormals(std::shared_ptr<Hittable> p) : ptr(p) {}

        virtual bool hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const override {
            if(ptr->hit(r, tMin, tMax, rec)) {
                rec.normal = -rec.normal;
                return true;
            }
            return false;
        }

        virtual bool boundingBox(double t0, double t1, AABB &outputBox) const override {
            return ptr->boundingBox(t0,t1,outputBox);
        }
    public:
        std::shared_ptr<Hittable> ptr;
};

#endif //RAYTRACER_FLIPNORMALS_H
