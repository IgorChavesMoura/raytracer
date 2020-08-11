//
// Created by igor on 06/08/2020.
//

#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H

#include "util.h"

class Camera {

    public:
        Camera(point3 lookfrom, point3 lookat, Vector3 vup, double vfov, double aspect_ratio, double aperture, double focus_dist){


            auto theta = degrees_to_radians(vfov);
            auto h = std::tan(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width  = aspect_ratio * viewport_height;

            auto focal_length = 1.0;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            len_radius = aperture / 2;


        }

        Ray get_ray(double s, double t) const {

            Vector3 rd = len_radius * random_in_unit_disk();
            Vector3 offset = u * rd.x() + v * rd.y();


            return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);

        }

    public:
        point3 origin;
        point3 lower_left_corner;

        Vector3 horizontal;
        Vector3 vertical;

        Vector3 u, v, w;

        double len_radius;
};


#endif //RAYTRACER_CAMERA_H
