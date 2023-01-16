//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_HITTABLE_H
#define RAYTRACER_HITTABLE_H

#include "../Ray.h"
#include "../AABB.h"

class Material;

struct RotationTransform {
    double theta; //radians
    double sinTheta,cosTheta;
};

struct TranslationTransform {
    Vector3 offset;
};

struct Transform {
    RotationTransform rotation;
    TranslationTransform translation;
};

struct HitRecord {

    Point3 p;
    Vector3 normal;
    std::shared_ptr<Material> matPtr;
    double t;
    double u;
    double v;
    bool frontFace;

    inline void setFaceNormal(const Ray& r, const Vector3& outward_normal){

        frontFace = dot(r.direction(), outward_normal) < 0;

        normal = frontFace ? outward_normal : -outward_normal;

    }

};

class Hittable {

    public:
        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const = 0;
        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const = 0;
};


#endif //RAYTRACER_HITTABLE_H
