#ifndef RAYTRACER_DiffuseLight_cuh
#define RAYTRACER_DiffuseLight_cuh
 
#include "Material.cuh"
#include "../Texture.cuh"
#include "../hittable/Hittable.cuh"

class DiffuseLight : public Material {
    public:
        __device__ DiffuseLight() {}
        __device__ DiffuseLight(Color c) : emit(new SolidColor(c)) {}
        __device__ DiffuseLight(SolidColor* e) : emit(e) {}

       __device__  virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const override {
            return false;
        }

        __device__ virtual Color emitted(float u, float v, const Point3& p) const override {
            return emit->value(u,v,p);
        }
    public:
        Texture* emit;
};

#endif // RAYTRACER_DiffuseLight_cuh