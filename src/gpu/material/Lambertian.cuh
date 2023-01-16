#ifndef RAYTRACER_Lambertian_cuh
#define RAYTRACER_Lambertian_cuh
 
#include "../util.cuh"

#include "Material.cuh"
#include "../hittable/Hittable.cuh"
#include "../Texture.cuh"
#include "../Ray.cuh"

class Lambertian : public Material {

    public:
        __device__ Lambertian(const Color& a) : albedo(new SolidColor(a)) {}
        __device__ Lambertian(Texture* a) : albedo(a) {}

        __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const override {
            Vector3 scatterDirection = rec.normal + randomUnitVector(randState);
            scattered = Ray(rec.p, scatterDirection, rIn.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);

            return true;
        }


    public:
        Texture* albedo;

};

 
#endif // RAYTRACER_Lambertian_cuh