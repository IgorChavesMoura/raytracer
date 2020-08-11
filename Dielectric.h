//
// Created by igor on 07/08/2020.
//

#ifndef RAYTRACER_DIELECTRIC_H
#define RAYTRACER_DIELECTRIC_H


#include "Material.h"
#include "Hittable.h"

double schlick(double cosine, double ref_idx){

    auto r0 = (1 - ref_idx) / (1 + ref_idx);

    r0 = r0*r0;

    return r0 + (1 - r0)*std::pow((1 - cosine), 5);

}


class Dielectric : public Material {

    public:
        Dielectric(double ri) : ref_idx(ri) {}

        virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {

            attenuation = color(1.0, 1.0, 1.0);

            double etai_over_etat = rec.front_face ? (1.0/ref_idx) : ref_idx;

            Vector3 unit_direction = unit_vector(r_in.direction());

            double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

            if(etai_over_etat * sin_theta > 1.0){

                Vector3 reflected = reflect(unit_direction, rec.normal);
                scattered = Ray(rec.p, reflected);

                return true;

            }

            double reflect_prob = schlick(cos_theta, etai_over_etat);

            if(random_double() < reflect_prob){

                Vector3 reflected = reflect(unit_direction, rec.normal);

                scattered = Ray(rec.p, reflected);

                return true;

            }

            Vector3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
            scattered = Ray(rec.p, refracted);

            return true;


        }


    public:
        double ref_idx;
};


#endif //RAYTRACER_DIELECTRIC_H
