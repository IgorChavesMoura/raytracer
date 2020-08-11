//
// Created by igor on 06/08/2020.
//

#ifndef RAYTRACER_MATERIAL_H
#define RAYTRACER_MATERIAL_H

#include "util.h"

struct hit_record;

class Material {

    public:
        virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const = 0;

};


#endif //RAYTRACER_MATERIAL_H
