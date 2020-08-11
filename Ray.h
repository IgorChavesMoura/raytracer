//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H

#include "Vector3.h"

class Ray {

    public:

        Ray() { }
        Ray(const point3& origin, const Vector3& direction) : orig(origin), dir(direction) { }

        point3 origin() const {

            return orig;

        }

        Vector3 direction() const {

            return dir;

        }

        point3 at(double t) const {

            return orig + t*dir;

        }


    public:
        point3 orig;
        Vector3 dir;

};


#endif //RAYTRACER_RAY_H
