//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_HITTABLELIST_H
#define RAYTRACER_HITTABLELIST_H

#include "Hittable.h"

#include <memory>
#include <vector>

class HittableList : public Hittable {

    public:

        HittableList() {}

        HittableList(std::shared_ptr<Hittable> object) {
            add(object);
        }

        void clear() {
            objects.clear();
        }

        void add(std::shared_ptr<Hittable> object) {
            objects.push_back(object);
        }

        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;
        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override;

    public:
        std::vector<std::shared_ptr<Hittable>> objects;
};

bool HittableList::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {

    HitRecord tempRec;
    bool hitAnything = false;
    auto closestSoFar = tMax;

    for(const auto& object : objects){
        if(object->hit(r, tMin, closestSoFar, tempRec)){

            hitAnything = true;
            closestSoFar = tempRec.t;
            rec = tempRec;

        }
    }

    return hitAnything;

}

bool HittableList::boundingBox(double t0, double t1, AABB &outputBox) const {
    if(objects.empty()) return false;

    AABB tempBox;
    bool firstBox = true;

    for(const auto& object : objects) {
        if(!object->boundingBox(t0,t1,tempBox)) return false;

        outputBox = firstBox ? tempBox : sorroundingBox(outputBox, tempBox);
        firstBox = false;
    }

    return true;
}

#endif //RAYTRACER_HITTABLELIST_H
