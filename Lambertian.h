//
// Created by igor on 07/08/2020.
//

#ifndef RAYTRACER_LAMBERTIAN_H
#define RAYTRACER_LAMBERTIAN_H

#include "util.h"

#include "Material.h"
#include "Hittable.h"

class Lambertian : public Material {

    public:
        Lambertian(const color& a) : albedo(a) {}

        virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {

            Vector3 scatter_direction = rec.normal + random_unit_vector();
            scattered = Ray(rec.p, scatter_direction);
            attenuation = albedo;

            return true;


        }


    public:
        color albedo;

};


#endif //RAYTRACER_LAMBERTIAN_H
