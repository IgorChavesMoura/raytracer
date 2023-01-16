#ifndef RAYTRACER_Camera_cuh
#define RAYTRACER_Camera_cuh

#include "Vector3.cuh" 
#include "util.cuh"

class Camera {

    public:
        __device__ Camera(Point3 lookfrom, Point3 lookat, Vector3 vup, float vfov, float aspectRatio, float aperture, float focusDist, float t0, float t1){

            auto theta = degreesToRadians(vfov);
            auto h = std::tan(theta/2);
            auto viewportHeight = 2.0f * h;
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

        __device__ Ray getRay(float s, float t, curandState* randState) const {

            Vector3 rd = lenRadius * randomInUnitDisk(randState);
            Vector3 offset = u * rd.x() + v * rd.y();


            return Ray(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - origin - offset,
                       randomFloat(randState, time0, time1));

        }

    public:
        Point3 origin;
        Point3 lowerLeftCorner;

        Vector3 horizontal;
        Vector3 vertical;

        Vector3 u, v, w;

        float lenRadius, time0, time1;
};
 
#endif // RAYTRACER_Camera_cuh