//
// Created by moura on 22/12/2022.
//

#ifndef RAYTRACER_BVHNODE_H
#define RAYTRACER_BVHNODE_H

#include <algorithm>

#include "hittable/Hittable.h"
#include "hittable/HittableList.h"

inline bool boxCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b, int axis) {
    AABB boxA;
    AABB boxB;

    if(!a->boundingBox(0,0,boxA) || !b->boundingBox(0,0,boxB))
        std::cerr << "No bounding box in BVHNode constructor.\n";

    return boxA.min().e[axis] < boxB.min().e[axis];
}

bool boxXCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return boxCompare(a,b,0);
}

bool boxYCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return boxCompare(a,b,1);
}

bool boxZCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return boxCompare(a,b,2);
}

class BVHNode : public Hittable {
    public:
        BVHNode();
        BVHNode(const HittableList& list, double t0, double t1) : BVHNode(list.objects, 0, list.objects.size(), t0, t1) {}
        BVHNode(const std::vector<std::shared_ptr<Hittable>>& srcObjects, size_t start, size_t end, double t0, double t1);

        virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override;

    public:
        std::shared_ptr<Hittable> left;
        std::shared_ptr<Hittable> right;
        AABB box;
};

BVHNode::BVHNode(const std::vector<std::shared_ptr<Hittable>> &srcObjects, size_t start, size_t end, double t0,
                 double t1) {
    auto objects = srcObjects; // Create a modifiable array of the source scene objects

    int axis = randomInt(0, 2);
    auto comparator = (axis == 0) ? boxXCompare : (axis == 1) ? boxYCompare : boxZCompare;

    size_t objectsSpan = end - start;

    if(objectsSpan == 1) {
        left = right = objects[start];
    } else if(objectsSpan == 2) {
        if(comparator(objects[start], objects[start+1])) {
            left = objects[start];
            right = objects[start+1];
        } else {
            left = objects[start+1];
            right = objects[start];
        }
    } else {
        std::sort(objects.begin() + start,objects.begin() + end,comparator);

        auto mid = start + objectsSpan/2;

        left = std::make_shared<BVHNode>(objects,start,mid,t0,t1);
        right = std::make_shared<BVHNode>(objects,mid,end,t0,t1);
    }

    AABB boxLeft, boxRight;

    if(!left->boundingBox(t0,t1,boxLeft) || !right->boundingBox(t0,t1,boxRight)) {
        std::cerr << "No bounding box in BVHNode constructor.\n";
    }

    box = sorroundingBox(boxLeft,boxRight);
}

bool BVHNode::hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const {
    if(!box.hit(r, t_min, t_max)) return false;

    HitRecord leftRec,rightRec;

    bool hitLeft = left->hit(r, t_min, t_max, leftRec);
    bool hitRight = right->hit(r, t_min, t_max, rightRec);

    if(hitLeft && hitRight) {
        if(leftRec.t < rightRec.t) {
            rec = leftRec;
        } else {
            rec = rightRec;
        }

        return true;
    } else if(hitLeft) {
        rec = leftRec;
        return true;
    } else if(hitRight) {
        rec = rightRec;
        return true;
    } else return false;
}

bool BVHNode::boundingBox(double t0, double t1, AABB &outputBox) const {
    outputBox = box;

    return true;
}

#endif //RAYTRACER_BVHNODE_H
