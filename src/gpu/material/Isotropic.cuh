#ifndef RAYTRACER_ISOTROPIC_CUH
#define RAYTRACER_ISOTROPIC_CUH

#include "Material.cuh"
#include "../hittable/Hittable.cuh"
#include "../Texture.cuh"
#include "../util.cuh"

class Isotropic : public Material {
    public:
        __device__ Isotropic(const Color& c) : albedo(new SolidColor(c)) {}
        __device__ Isotropic(Texture* a) : albedo(a) {}

        __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const override {
            scattered = Ray(rec.p, randomInUnitSphere(randState), rIn.time());
            attenuation = albedo->value(rec.u,rec.v,rec.p);

            return true;
        }
    public:
        Texture* albedo;
};

#endif //RAYTRACER_ISOTROPIC_CUH
