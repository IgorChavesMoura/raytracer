//
// Created by moura on 28/12/2022.
//

#ifndef RAYTRACER_ISOTROPIC_H
#define RAYTRACER_ISOTROPIC_H

#include "Material.h"
#include "../hittable/Hittable.h"
#include "../Texture.h"

class Isotropic : public Material {
    public:
        Isotropic(Color c) : albedo(std::make_shared<SolidColor>(c)) {}
        Isotropic(std::shared_ptr<Texture> a) : albedo(a) {}

        virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
            scattered = Ray(rec.p, randomInUnitSphere(), rIn.time());
            attenuation = albedo->value(rec.u,rec.v,rec.p);

            return true;
        }
    public:
        std::shared_ptr<Texture> albedo;
};

#endif //RAYTRACER_ISOTROPIC_H
