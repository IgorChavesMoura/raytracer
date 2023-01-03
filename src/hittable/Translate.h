//
// Created by moura on 27/12/2022.
//

#ifndef RAYTRACER_TRANSLATE_H
#define RAYTRACER_TRANSLATE_H

#include "Hittable.h"
#include "../AABB.h"

class Translate : public Hittable {
    public:
        Translate(std::shared_ptr<Hittable> p, const Vector3& displacement) : ptr(p), offset(displacement)  {}

        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;

        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override;
    public:
        std::shared_ptr<Hittable> ptr;
        Vector3 offset;
};

bool Translate::hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const {
    Ray movedRay(r.origin() - offset, r.direction(), r.time());

    if(!ptr->hit(movedRay, tMin, tMax, rec)) return false;

    rec.p += offset;

    return true;
}

bool Translate::boundingBox(double t0, double t1, AABB &outputBox) const {
    if(!ptr->boundingBox(t0,t1,outputBox)) return false;

    outputBox = AABB(outputBox.min() + offset, outputBox.max() + offset);

    return true;
}

#endif //RAYTRACER_TRANSLATE_H
