//
// Created by igor on 08/08/2020.
//

#ifndef RAYTRACER_TRIANGLE_H
#define RAYTRACER_TRIANGLE_H

#include "Hittable.h"
#include "Vector3.h"


class Triangle : public Hittable {

    public:
        Triangle() { }
        Triangle(Vector3 vp0, Vector3 vp1, Vector3 vp2, std::shared_ptr<Material> m) : v0(vp0), v1(vp1), v2(vp2), material(m){

            normal = cross(v1 - v0, v2 - v0);

        }


        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;


    public:
        Vector3 v0, v1, v2;

        std::shared_ptr<Material> material;

        Vector3 normal;


};

bool Triangle::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {

    double theta, t, u, v;

    Vector3 v0v1 = v1 - v0;
    Vector3 v0v2 = v2 - v0;

    Vector3 pvec = cross(r.direction(), v0v2);

    double det = dot(pvec, v0v1);
    double kEpsilon = 0.00001;

    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if(det < kEpsilon){

        return false;

    }

    double invDet = 1 / det;

    Vector3 tvec = r.origin() - v0;
    u = dot(tvec,pvec) * invDet;

    if(u < 0 || u > 1) {

        return false;

    }

    Vector3 qvec =  cross(tvec, v0v1);
    v = dot(r.direction(), qvec) * invDet;

    if(v < 0 || u + v > 1){

        return false;

    }


    t = dot(v0v2, qvec) * invDet;

    if(t < 0){

        return false;

    }

    rec.p = r.at(t);

    rec.t = t;
    rec.normal = normal;
    rec.mat_ptr = material;

    return true;

}


#endif //RAYTRACER_TRIANGLE_H
