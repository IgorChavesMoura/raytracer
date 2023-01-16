#ifndef RAYTRACER_Material_cuh
#define RAYTRACER_Material_cuh
 
#include "../Vector3.cuh"

struct HitRecord;

class Material {

    public:
        __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const = 0;
        __device__ virtual Color emitted(float u, float v, const Point3& p) const {
            return Color(0.0f, 0.0f, 0.0f); //Default behaviour
        }
};
 
#endif // RAYTRACER_Material_cuh