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

        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override;

        void rotate(double angle);
        void translate(Vector3 translation);

    public:
        Point3 boxMin;
        Point3 boxMax;
        HittableList sides;
        Transform transform;
};

Box::Box(const Point3 &p0, const Point3 &p1, std::shared_ptr<Material> ptr) {
    boxMin = p0;
    boxMax = p1;

    transform.translation.offset = Vector3(0.0, 0.0, 0.0);

    transform.rotation.theta = 0.0;
    transform.rotation.cosTheta = 1.0;
    transform.rotation.sinTheta = 0.0;

    sides.add(std::make_shared<XYRect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
    sides.add(std::make_shared<XYRect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

    sides.add(std::make_shared<XZRect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
    sides.add(std::make_shared<XZRect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

    sides.add(std::make_shared<YZRect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
    sides.add(std::make_shared<YZRect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

void Box::rotate(double angle) {
    transform.rotation.theta = degreesToRadians(angle);
    transform.rotation.sinTheta = sin(transform.rotation.theta);
    transform.rotation.cosTheta = cos(transform.rotation.theta);
}

void Box::translate(Vector3 translation) {
    transform.translation.offset = translation;
}

bool Box::hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const {
    Ray movedRay(r.origin() - transform.translation.offset, r.direction(), r.time());
    
    if(transform.rotation.theta == 0.0) {
        bool hit = sides.hit(movedRay, tMin, tMax, rec);
        
        if(hit) {
            rec.p += transform.translation.offset;
        }

        return hit;
    };

    float cosTheta = transform.rotation.cosTheta;
    float sinTheta = transform.rotation.sinTheta;

    auto origin = movedRay.origin();
    auto direction = movedRay.direction();
    
    origin[0] = cosTheta*movedRay.origin()[0] - sinTheta*movedRay.origin()[2];
    origin[2] = sinTheta*movedRay.origin()[0] + cosTheta*movedRay.origin()[2];

    direction[0] = cosTheta*movedRay.direction()[0] - sinTheta*movedRay.direction()[2];
    direction[2] = sinTheta*movedRay.direction()[0] + cosTheta*movedRay.direction()[2];

    Ray rotatedR(origin,direction,movedRay.time());

    if(!sides.hit(rotatedR,tMin,tMax,rec)) return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] = cosTheta*rec.p[0] + sinTheta*rec.p[2];
    p[2] = -sinTheta*rec.p[0] + cosTheta*rec.p[2];

    normal[0] = cosTheta*rec.normal[0] + sinTheta*rec.normal[2];
    normal[2] = -sinTheta*rec.normal[0] + cosTheta*rec.normal[2];

    rec.p = p + transform.translation.offset;
    rec.setFaceNormal(rotatedR, normal);

    return true;
}

bool Box::boundingBox(double t0, double t1, AABB& outputBox) const {
    AABB bBox = AABB(boxMin, boxMax);
    
    if(transform.rotation.theta == 0.0) {
        outputBox = AABB(bBox.min() + transform.translation.offset, bBox.max() + transform.translation.offset);
        return true;
    }

    double cosTheta = transform.rotation.cosTheta;
    double sinTheta = transform.rotation.sinTheta;

    double infinity = INFINITY;

    Point3 min(infinity, infinity, infinity);
    Point3 max(-infinity, -infinity, -infinity);

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) {
                auto x = i*bBox.max().x() + (1-i)*bBox.min().x();
                auto y = j*bBox.max().y() + (1-j)*bBox.min().y();
                auto z = k*bBox.max().z() + (1-k)*bBox.min().z();

                auto newx =  cosTheta*x + sinTheta*z;
                auto newz = -sinTheta*x + cosTheta*z;

                Vector3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {

                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bBox = AABB(min + transform.translation.offset,max + transform.translation.offset);

    outputBox = bBox;

    return true;
}

#endif //RAYTRACER_BOX_H
