#ifndef RAYTRACER_Hittable_cuh
#define RAYTRACER_Hittable_cuh
 
#include "../Ray.cuh"
#include "../AABB.cuh"


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
    Material* matPtr;
    float t;
    float u;
    float v;
    bool frontFace;

    __device__ inline void setFaceNormal(const Ray& r, const Vector3& outward_normal){

        frontFace = dot(r.direction(), outward_normal) < 0;

        normal = frontFace ? outward_normal : -outward_normal;

    }

};

class Hittable {

    public:
        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const = 0;
        __device__ virtual bool boundingBox(float t0, float t1, AABB& outputBox) const = 0;
};

 
#endif // RAYTRACER_Hittable_cuh