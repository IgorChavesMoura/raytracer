//
// Created by igor on 07/08/2020.
//

#ifndef RAYTRACER_DIELECTRIC_H
#define RAYTRACER_DIELECTRIC_H


#include "Material.h"
#include "../hittable/Hittable.h"

double schlick(double cosine, double refIdx){

    auto r0 = (1 - refIdx) / (1 + refIdx);

    r0 = r0*r0;

    return r0 + (1 - r0)*std::pow((1 - cosine), 5);

}


class Dielectric : public Material {

    public:
        Dielectric(double ri) : refIdx(ri) {}

        virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {

            attenuation = Color(1.0, 1.0, 1.0);

            double etaiOverEtat = rec.frontFace ? (1.0 / refIdx) : refIdx;

            Vector3 unitDirection = unitVector(rIn.direction());

            double cosTheta = std::fmin(dot(-unitDirection, rec.normal), 1.0);
            double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

            if(etaiOverEtat * sinTheta > 1.0){

                Vector3 reflected = reflect(unitDirection, rec.normal);
                scattered = Ray(rec.p, reflected);

                return true;

            }

            double reflectProb = schlick(cosTheta, etaiOverEtat);

            if(randomDouble() < reflectProb){

                Vector3 reflected = reflect(unitDirection, rec.normal);

                scattered = Ray(rec.p, reflected);

                return true;

            }

            Vector3 refracted = refract(unitDirection, rec.normal, etaiOverEtat);
            scattered = Ray(rec.p, refracted, rIn.time());

            return true;


        }


    public:
        double refIdx;
};


#endif //RAYTRACER_DIELECTRIC_H
