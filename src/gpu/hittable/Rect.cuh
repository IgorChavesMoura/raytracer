#ifndef RAYTRACER_Rect_cuh
#define RAYTRACER_Rect_cuh
 
#include "Hittable.cuh"
#include "../material/Material.cuh"

class YZRect : public Hittable {
    public:
        __device__ YZRect() {}
        __device__ YZRect(float _y0, float _y1, float _z0, float _z1, float _k, Material* material) 
                    : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(material) {}

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const override;
        __device__ virtual bool boundingBox(float t0, float t1, AABB& outputBox) const override {

            // Making some thickness on bounding box x component
            auto x0 = k - 0.0001f;
            auto x1 = k + 0.0001f;

            outputBox = AABB(Point3(x0, y0, z0), Point3(x1, y1, z1));

            return true;
        }


    private:
        float y0,y1,z0,z1,k;
        Material* mp;
};

class XZRect : public Hittable {
    public:
        __device__ XZRect() {}
        __device__ XZRect(float _x0, float _x1, float _z0, float _z1, float _k, Material* material) 
                    : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(material) {}

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const override;
        __device__ virtual bool boundingBox(float t0, float t1, AABB& outputBox) const override {

            // Making some thickness on bounding box y component
            auto y0 = k - 0.0001f;
            auto y1 = k + 0.0001f;

            outputBox = AABB(Point3(x0, y0, z0), Point3(x1, y1, z1));

            return true;
        }


    private:
        float x0,x1,z0,z1,k;
        Material* mp;
};

class XYRect : public Hittable {
    public:
        __device__ XYRect() {}
        __device__ XYRect(float _x0, float _x1, float _y0, float _y1, float _k, Material* material) 
                    : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(material) {}

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const override;
        __device__ virtual bool boundingBox(float t0, float t1, AABB& outputBox) const override {

            // Making some thickness on bounding box z component
            auto z0 = k - 0.0001f;
            auto z1 = k + 0.0001f;

            outputBox = AABB(Point3(x0, y0, z0), Point3(x1, y1, z1));

            return true;
        }


    private:
        float x0,x1,y0,y1,k;
        Material* mp;
};

__device__ bool YZRect::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const {
    auto t = (k - r.origin().x())/r.direction().x();

    if(t < tMin || t > tMax) return false;

    auto y = r.origin().y() + (t * r.direction().y());
    auto z = r.origin().z() + (t * r.direction().z());

    if((y < y0 || y > y1) || (z < z0 || z > z1)) return false;

    rec.u = (y-y0)/(y1-y0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;

    auto outwardNormal = Vector3(1,0,0);
    rec.setFaceNormal(r, outwardNormal);
    rec.matPtr = mp;
    rec.p = r.at(t);

    return true;
}

__device__ bool XZRect::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const {
    auto t = (k - r.origin().y())/r.direction().y();

    if(t < tMin || t > tMax) return false;

    auto x = r.origin().x() + (t * r.direction().x());
    auto z = r.origin().z() + (t * r.direction().z());

    if((x < x0 || x > x1) || (z < z0 || z > z1)) return false;

    rec.u = (x-x0)/(x1-x0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;

    auto outwardNormal = Vector3(0,1,0);
    rec.setFaceNormal(r, outwardNormal);
    rec.matPtr = mp;
    rec.p = r.at(t);

    return true;

}

__device__ bool XYRect::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const {
    auto originZ = r.origin().z();
    auto directionZ = r.direction().z();
    auto t = (k - originZ)/directionZ;

    if(t < tMin || t > tMax) return false;

    auto x = r.origin().x() + (t * r.direction().x());
    auto y = r.origin().y() + (t * r.direction().y());

    if((x < x0 || x > x1) || (y < y0 || y > y1)) return false;

    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;

    auto outwardNormal = Vector3(0,0,1);
    rec.setFaceNormal(r, outwardNormal);
    rec.matPtr = mp;
    rec.p = r.at(t);

    return true;

}
 
#endif // RAYTRACER_Rect_cuh