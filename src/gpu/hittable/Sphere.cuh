#ifndef RAYTRACER_Sphere_cuh
#define RAYTRACER_Sphere_cuh
 
#include "Hittable.cuh"
#include "../Vector3.cuh"
 
class Sphere : public Hittable {

    public:
        __device__ Sphere() { }
        __device__ Sphere(Point3 cen, float r, Material* m) : center(cen), radius(r), matPtr(m) {};

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const override;
        __device__ virtual bool boundingBox(float t0, float t1, AABB& outputBox) const override;

    public:
        Point3 center;
        float radius;
        Material* matPtr;

    private:
        __device__ static void getSphereUv(const Point3& p, float& u, float& v) {
            auto theta = acos(-p.y());
            auto phi = atan2(-p.z(), p.x()) + PI;

            u = phi/(2*PI);
            v = theta/PI;
        }
};

__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const {

    Vector3 oc = r.origin() - center;

    auto a = r.direction().length_squared();
    auto halfB = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = halfB * halfB - a * c;

    if(discriminant > 0){

        auto root = std::sqrt(discriminant);

        auto temp = (-halfB - root) / a;

        if(temp < tMax && temp > tMin){

            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(r, outward_normal);
            getSphereUv(outward_normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            return true;
        }

        temp = (-halfB + root) / a;

        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(r, outward_normal);
            getSphereUv(outward_normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            return true;
        }

    }

    return false;

}

__device__ bool Sphere::boundingBox(float t0, float t1, AABB &outputBox) const {
    outputBox = AABB(center - Vector3(radius,radius,radius), center + Vector3(radius,radius,radius));

    return true;
}


#endif // RAYTRACER_Sphere_cuh