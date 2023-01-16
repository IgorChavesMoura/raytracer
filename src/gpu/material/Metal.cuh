#ifndef RAYTRACER_METAL_CUH
#define RAYTRACER_METAL_CUH

#include "Material.cuh"
#include "../hittable/Hittable.cuh"
#include "../util.cuh"

class Metal : public Material {

    public:
        __device__ Metal(const Color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const override {

            Vector3 reflected = reflect(unitVector(rIn.direction()), rec.normal);

            scattered = Ray(rec.p, reflected + fuzz * randomInUnitSphere(randState), rIn.time());

            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0);


        }

    public:
        Color albedo;
        float fuzz;

};


#endif //RAYTRACER_METAL_CUH
