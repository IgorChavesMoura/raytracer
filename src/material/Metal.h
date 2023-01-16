//
// Created by igor on 07/08/2020.
//

#ifndef RAYTRACER_METAL_H
#define RAYTRACER_METAL_H

#include "Material.h"
#include "../hittable/Hittable.h"

class Metal : public Material {

    public:
        Metal(const Color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {

            Vector3 reflected = reflect(unitVector(rIn.direction()), rec.normal);

            scattered = Ray(rec.p, reflected + fuzz * randomInUnitSphere(), rIn.time());

            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0);


        }

    public:
        Color albedo;
        double fuzz;

};


#endif //RAYTRACER_METAL_H
 