//
// Created by igor on 06/08/2020.
//

#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H

#include "util.h"

class Camera {

    public:
        Camera(Point3 lookfrom, Point3 lookat, Vector3 vup, double vfov, double aspectRatio, double aperture, double focusDist, double t0, double t1){


            auto theta = degreesToRadians(vfov);
            auto h = std::tan(theta/2);
            auto viewportHeight = 2.0 * h;
            auto viewportWidth  = aspectRatio * viewportHeight;

            w = unitVector(lookfrom - lookat);
            u = unitVector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focusDist * viewportWidth * u;
            vertical = focusDist * viewportHeight * v;
            lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - focusDist * w;

            lenRadius = aperture / 2;

            time0 = t0;
            time1 = t1;


        }

        Ray getRay(double s, double t) const {

            Vector3 rd = lenRadius * randomInUnitDisk();
            Vector3 offset = u * rd.x() + v * rd.y();


            return Ray(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - origin - offset,
                       randomDouble(time0, time1));

        }

    public:
        Point3 origin;
        Point3 lowerLeftCorner;

        Vector3 horizontal;
        Vector3 vertical;

        Vector3 u, v, w;

        double lenRadius, time0, time1;
};


#endif //RAYTRACER_CAMERA_H
