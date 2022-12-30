//
// Created by moura on 27/12/2022.
//

#ifndef RAYTRACER_ROTATE_H
#define RAYTRACER_ROTATE_H

#include "Hittable.h"
#include "../AABB.h"

#include "../util.h"

class RotateY : public Hittable {
    public:
        RotateY(std::shared_ptr<Hittable> p, double angle);

        virtual bool hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const override;

        virtual bool boundingBox(double t0, double t1, AABB &outputBox) const override {
            outputBox = bBox;

            return hasBox;
        }

    public:
        std::shared_ptr<Hittable> ptr;
        double sinTheta, cosTheta;
        bool hasBox;
        AABB bBox;
};

RotateY::RotateY(std::shared_ptr<Hittable> p, double angle) : ptr(p) {
    auto radians = degreesToRadians(angle);

    sinTheta = sin(radians);
    cosTheta = cos(radians);
    hasBox = ptr->boundingBox(0,1,bBox);

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

    bBox = AABB(min,max);
}

bool RotateY::hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const {
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cosTheta*r.origin()[0] - sinTheta*r.origin()[2];
    origin[2] = sinTheta*r.origin()[0] + cosTheta*r.origin()[2];

    direction[0] = cosTheta*r.direction()[0] - sinTheta*r.direction()[2];
    direction[2] = sinTheta*r.direction()[0] + cosTheta*r.direction()[2];

    Ray rotatedR(origin,direction,r.time());

    if(!ptr->hit(rotatedR,t_min,t_max,rec)) return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] = cosTheta*rec.p[0] + sinTheta*rec.p[2];
    p[2] = -sinTheta*rec.p[0] + cosTheta*rec.p[2];

    normal[0] = cosTheta*rec.normal[0] + sinTheta*rec.normal[2];
    normal[2] = -sinTheta*rec.normal[0] + cosTheta*rec.normal[2];

    rec.p = p;
    rec.setFaceNormal(rotatedR, normal);

    return true;
}

#endif //RAYTRACER_ROTATE_H
