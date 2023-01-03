//
// Created by moura on 26/12/2022.
//

#ifndef RAYTRACER_RECT_H
#define RAYTRACER_RECT_H

#include "Hittable.h"
#include "../material/Material.h"

class XYRect : public Hittable {
    public:
        XYRect() {}
        XYRect(double _x0, double _x1, double _y0, double _y1, double _k, std::shared_ptr<Material> material)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(material) {}


        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;
        virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override {

            // Making some thickness on bounding box z component
            auto z0 = k - 0.0001;
            auto z1 = k + 0.0001;

            outputBox = AABB(Point3(x0, y0, z0), Point3(x1, y1, z1));

            return true;
        }

    private:
        double x0,x1,y0,y1,k;
        std::shared_ptr<Material> mp;
};

class XZRect : public Hittable {
public:
    XZRect() {}
    XZRect(double _x0, double _x1, double _z0, double _z1, double _k, std::shared_ptr<Material> material)
            : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(material) {}


    virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;
    virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override {

        // Making some thickness on bounding box z component
        auto y0 = k - 0.0001;
        auto y1 = k + 0.0001;

        outputBox = AABB(Point3(x0, y0, z0), Point3(x1, y1, z1));

        return true;
    }

private:
    double x0,x1,z0,z1,k;
    std::shared_ptr<Material> mp;
};

class YZRect : public Hittable {
public:
    YZRect() {}
    YZRect(double _y0, double _y1, double _z0, double _z1, double _k, std::shared_ptr<Material> material)
            : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(material) {}


    virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;
    virtual bool boundingBox(double t0, double t1, AABB& outputBox) const override {

        // Making some thickness on bounding box z component
        auto x0 = k - 0.0001;
        auto x1 = k + 0.0001;

        outputBox = AABB(Point3(x0, y0, z0), Point3(x1, y1, z1));

        return true;
    }

private:
    double y0,y1,z0,z1,k;
    std::shared_ptr<Material> mp;
};


bool XYRect::hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const {
    auto t = (k - r.origin().z())/r.direction().z();

    if(t < tMin || t > tMax) return false;

    auto x = r.origin().x() + (t * r.direction().x());
    auto y = r.origin().y() + (t * r.direction().y());

    if((x < x0 || x > x1) || (y < y0 || y > y1)) return false;

    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;

    auto outward_normal = Vector3(0,0,1);
    rec.setFaceNormal(r, outward_normal);
    rec.matPtr = mp;
    rec.p = r.at(t);

    return true;
}

bool XZRect::hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const {
    auto t = (k - r.origin().y())/r.direction().y();

    if(t < tMin || t > tMax) return false;

    auto x = r.origin().x() + (t * r.direction().x());
    auto z = r.origin().z() + (t * r.direction().z());

    if((x < x0 || x > x1) || (z < z0 || z > z1)) return false;

    rec.u = (x-x0)/(x1-x0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;

    auto outward_normal = Vector3(0,1,0);
    rec.setFaceNormal(r, outward_normal);
    rec.matPtr = mp;
    rec.p = r.at(t);

    return true;
}

bool YZRect::hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const {
    auto t = (k - r.origin().x())/r.direction().x();

    if(t < tMin || t > tMax) return false;

    auto y = r.origin().y() + (t * r.direction().y());
    auto z = r.origin().z() + (t * r.direction().z());

    if((y < y0 || y > y1) || (z < z0 || z > z1)) return false;

    rec.u = (y-y0)/(y1-y0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;

    auto outward_normal = Vector3(1,0,0);
    rec.setFaceNormal(r, outward_normal);
    rec.matPtr = mp;
    rec.p = r.at(t);

    return true;
}

#endif //RAYTRACER_RECT_H
