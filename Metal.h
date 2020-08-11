//
// Created by igor on 07/08/2020.
//

#ifndef RAYTRACER_METAL_H
#define RAYTRACER_METAL_H

#include "Material.h"
#include "Hittable.h"

class Metal : public Material {

    public:
        Metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {

            Vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

            scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere());

            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0);


        }

    public:
        color albedo;
        double fuzz;

};


#endif //RAYTRACER_METAL_H
