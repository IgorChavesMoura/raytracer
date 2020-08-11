cmake_minimum_required(VERSION 3.17)
project(RayTracer)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(RayTracer
        main.cc Vector3.h color.h Ray.h Hittable.h Sphere.h HittableList.h util.h Camera.h Material.h Lambertian.h Metal.h Dielectric.h Triangle.h Mesh.h FileParser.h)
