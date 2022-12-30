//
// Created by igor on 06/08/2020.
//

#ifndef RAYTRACER_MATERIAL_H
#define RAYTRACER_MATERIAL_H

#include "../util.h"

struct HitRecord;

class Material {

    public:
        virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const = 0;
        virtual Color emitted(double u, double v, const Point3& p) const {
            return Color(0, 0, 0); //Default behaviour
        }
};


#endif //RAYTRACER_MATERIAL_H
