#ifndef RAYTRACER_DIELECTRIC_CUH
#define RAYTRACER_DIELECTRIC_CUH


#include "Material.cuh"
#include "../hittable/Hittable.cuh"
#include "../util.cuh"

__device__ float schlick(float cosine, float refIdx){

    auto r0 = (1 - refIdx) / (1 + refIdx);

    r0 = r0*r0;

    return r0 + (1 - r0)*std::pow((1 - cosine), 5);

}


class Dielectric : public Material {

    public:
        __device__ Dielectric(float ri) : refIdx(ri) {}

        __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const override {

            attenuation = Color(1.0f, 1.0f, 1.0f);

            float etaiOverEtat = rec.frontFace ? (1.0f / refIdx) : refIdx;

            Vector3 unitDirection = unitVector(rIn.direction());

            float cosTheta = std::fmin(dot(-unitDirection, rec.normal), 1.0f);
            float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

            if(etaiOverEtat * sinTheta > 1.0f){

                Vector3 reflected = reflect(unitDirection, rec.normal);
                scattered = Ray(rec.p, reflected);

                return true;

            }

            float reflectProb = schlick(cosTheta, etaiOverEtat);

            if(randomFloat(randState) < reflectProb){

                Vector3 reflected = reflect(unitDirection, rec.normal);

                scattered = Ray(rec.p, reflected);

                return true;

            }

            Vector3 refracted = refract(unitDirection, rec.normal, etaiOverEtat);
            scattered = Ray(rec.p, refracted, rIn.time());

            return true;


        }


    public:
        float refIdx;
};


#endif //RAYTRACER_DIELECTRIC_CUH
