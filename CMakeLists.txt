cmake_minimum_required(VERSION 3.17)
project(RayTracer)

set(CMAKE_CXX_STANDARD 14)

include_directories(., material)

add_executable(RayTracer
        main.cc Vector3.h color.h Ray.h hittable/Hittable.h AABB.h hittable/Sphere.h hittable/HittableList.h BVHNode.h util.h Camera.h material/Material.h material/Lambertian.h material/Metal.h material/Dielectric.h material/DiffuseLight.h Texture.h hittable/Triangle.h hittable/MovingSphere.h hittable/Rect.h hittable/Translate.h)
