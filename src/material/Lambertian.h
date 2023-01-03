//
// Created by igor on 07/08/2020.
//

#ifndef RAYTRACER_LAMBERTIAN_H
#define RAYTRACER_LAMBERTIAN_H

#include "../util.h"

#include "Material.h"
#include "../hittable/Hittable.h"
#include "../Texture.h"

class Lambertian : public Material {

    public:
        Lambertian(const Color& a) : albedo(std::make_shared<SolidColor>(a)) {}
        Lambertian(std::shared_ptr<Texture> a) : albedo(a) {}

        virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
            Vector3 scatterDirection = rec.normal + randomUnitVector();
            scattered = Ray(rec.p, scatterDirection, rIn.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);

            return true;
        }


    public:
        std::shared_ptr<Texture> albedo;

};


#endif //RAYTRACER_LAMBERTIAN_H
