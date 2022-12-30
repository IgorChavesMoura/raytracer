//
// Created by moura on 27/12/2022.
//

#ifndef RAYTRACER_BOX_H
#define RAYTRACER_BOX_H

#include "Hittable.h"
#include "HittableList.h"
#include "Rect.h"
#include "../AABB.h"

class Box : public Hittable {
    public:
        Box() {}
        Box(const Point3& p0, const Point3& p1, std::shared_ptr<Material> ptr);

        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;

        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override {
            outputBox = AABB(boxMin, boxMax);

            return true;
        }

    public:
        Point3 boxMin;
        Point3 boxMax;
        HittableList sides;
};

Box::Box(const Point3 &p0, const Point3 &p1, std::shared_ptr<Material> ptr) {
    boxMin = p0;
    boxMax = p1;

    sides.add(std::make_shared<XYRect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
    sides.add(std::make_shared<XYRect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

    sides.add(std::make_shared<XZRect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
    sides.add(std::make_shared<XZRect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

    sides.add(std::make_shared<YZRect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
    sides.add(std::make_shared<YZRect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

bool Box::hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const {
    return sides.hit(r, tMin, tMax, rec);
}

#endif //RAYTRACER_BOX_H
