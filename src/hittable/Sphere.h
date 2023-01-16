//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_SPHERE_H
#define RAYTRACER_SPHERE_H

#include "Hittable.h"
#include "../Vector3.h"

class Sphere : public Hittable {

    public:
        Sphere() { }
        Sphere(Point3 cen, double r, std::shared_ptr<Material> m) : center(cen), radius(r), matPtr(m) {
            transform.translation.offset = Vector3(0.0,0.0,0.0);
            transform.rotation.theta = 0.0;
            transform.rotation.cosTheta = 1.0;
            transform.rotation.sinTheta = 0.0;
        };

        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;
        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override;

        void rotate(double angle);
        void translate(Vector3 translation);
        

    public:
        Point3 center;
        double radius;
        std::shared_ptr<Material> matPtr;
        Transform transform;

    private:
        void getSphereUv(const Point3& p, double& u, double& v) const {
            auto theta = acos(-p.y());
            auto phi = atan2(-p.z(), p.x()) + pi;

            u = (phi + transform.rotation.theta)/(2*pi);
            v = theta/pi;
        }
};

void Sphere::rotate(double angle) {
    transform.rotation.theta = degreesToRadians(angle);
    transform.rotation.sinTheta = sin(transform.rotation.theta);
    transform.rotation.cosTheta = cos(transform.rotation.theta);
}

void Sphere::translate(Vector3 translation) {
    transform.translation.offset = translation;
}


bool Sphere::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {
    Ray movedRay(r.origin() - transform.translation.offset, r.direction(), r.time());
    Vector3 oc = movedRay.origin() - center;

    auto a = movedRay.direction().length_squared();
    auto halfB = dot(oc, movedRay.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = halfB * halfB - a * c;

    if(discriminant > 0){

        auto root = std::sqrt(discriminant);

        auto temp = (-halfB - root) / a;

        if(temp < tMax && temp > tMin){

            rec.t = temp;
            rec.p = movedRay.at(rec.t);
            Vector3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(movedRay, outward_normal);
            getSphereUv(outward_normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            rec.p += transform.translation.offset;
            return true;
        }

        temp = (-halfB + root) / a;

        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = movedRay.at(rec.t);
            Vector3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(movedRay, outward_normal);
            getSphereUv(outward_normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            rec.p += transform.translation.offset;
            return true;
        }

    }

    return false;

}

bool Sphere::boundingBox(double t0, double t1, AABB &outputBox) const {
    outputBox = AABB(center - Vector3(radius,radius,radius) + transform.translation.offset, center + Vector3(radius,radius,radius) + transform.translation.offset);

    return true;
}


#endif //RAYTRACER_SPHERE_H
