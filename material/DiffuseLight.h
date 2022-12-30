//
// Created by moura on 26/12/2022.
//

#ifndef RAYTRACER_DIFFUSELIGHT_H
#define RAYTRACER_DIFFUSELIGHT_H

#include "Material.h"
#include "../Texture.h"
#include "../hittable/Hittable.h"

class DiffuseLight : public Material {
    public:
        DiffuseLight() {}
        DiffuseLight(Color c) : emit(std::make_shared<SolidColor>(c)) {}

        virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
            return false;
        }

        virtual Color emitted(double u, double v, const Point3& p) const override {
            return emit->value(u,v,p);
        }
    public:
        std::shared_ptr<Texture> emit;
};

#endif //RAYTRACER_DIFFUSELIGHT_H
