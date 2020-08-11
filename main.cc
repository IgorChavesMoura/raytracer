#include <iostream>

#include "util.h"
#include "color.h"
#include "FileParser.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Mesh.h"
#include "Camera.h"
#include "Material.h"
#include "Lambertian.h"
#include "Metal.h"
#include "Dielectric.h"

color ray_color(const Ray& r, const Hittable& world, int depth){

    hit_record rec;

    if(depth <= 0){

        return color(0,0,0);

    }

    if(world.hit(r,0.001,infinity,rec)) {

        Ray scattered;
        color attenuation;

        if(rec.mat_ptr->scatter(r,rec,attenuation,scattered)){

            return attenuation * ray_color(scattered,world,depth - 1);

        }

        return color(0,0,0);

    }


    Vector3 unit_direction = unit_vector(r.direction());

    auto t = 0.5*(unit_direction.y() + 1.0);

    return (1.0 - t)*color(1.0,1.0,1.0) + t*color(0.5,0.7,1.0);

}

int main(int argc, char** argv){

    //Image
    const auto ASPECT_RATIO = (3.0/2.0);
    const int IMAGE_WIDTH =  1200;
    const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
    const int SAMPLES_PER_PIXEL = 500;
    const int MAX_DEPTH = 50;

    //World
    HittableList world;

    auto material_ground = std::make_shared<Lambertian>(color(0.5, 0.5, 0.5));;


    world.add(std::make_shared<Sphere>(point3(0, -1000.5, -1), 1000, material_ground));



    auto albedo = color::random() * color::random();

    std::shared_ptr<Material> mat = std::make_shared<Lambertian>(albedo);

    std::shared_ptr<Mesh> ps5 = FileParser::parseStlFile("/home/igor/Downloads/Rack.stl");

    ps5->translate(Vector3(100, -160, 0));

    world.add(ps5);

//    std::shared_ptr<Mesh> cube = std::make_shared<Mesh>(mat);
//
//    cube->add_face(Vector3(-0.5, 1, -1), Vector3(-0.5, 0, -1), Vector3(0.5, 0, -1));
//    cube->add_face(Vector3(-0.5, 1, -1), Vector3(0.5, 0, -1), Vector3(0.5, 1, -1));
//
//    world.add(cube);

    //std::shared_ptr<Material> triangle_material = std::make_shared<Lambertian>(albedo);
//    world.add(std::make_shared<Triangle>(Vector3(-0.5,1,-1), Vector3(-0.5,0,-1), Vector3(0.5, 0, -1), mat));
//    world.add(std::make_shared<Triangle>(Vector3(-0.5,1,-1), Vector3(0.5,0,-1), Vector3(0.5,1,-1), mat));

    for(int a = -11; a < 11; a++){

        for(int b = -11; b < 11; b++){

            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if((center - point3(4, 0.2, 0)).length() > 0.9){

                std::shared_ptr<Material> sphere_material;

                if(choose_mat < 0.8){

                    auto albedo = color::random() * color::random();

                    sphere_material = std::make_shared<Lambertian>(albedo);

                    world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));

                } else if(choose_mat < 0.95){

                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);

                    sphere_material = std::make_shared<Metal>(albedo, fuzz);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));

                } else {

                    sphere_material = std::make_shared<Dielectric>(1.5);

                    world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));

                }

            }


        }

    }

    //Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    Vector3 vup(0, 1, 0);

    auto dist_to_focus = (lookfrom - lookat).length();
    auto aperture = 0.1;

    Camera cam(lookfrom, lookat, vup, 20, ASPECT_RATIO, aperture, dist_to_focus);

    //Render
    std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";

    for(int i = (IMAGE_HEIGHT - 1); i >= 0; i--){
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::endl << std::flush;
        for(int j = 0; j < IMAGE_WIDTH; j++){ 

            color pixel_color(0,0,0);

            for(int s = 0; s < SAMPLES_PER_PIXEL; s++){

                auto u = (j + random_double()) / (IMAGE_WIDTH - 1);
                auto v = (i + random_double()) / (IMAGE_HEIGHT - 1);

                Ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, MAX_DEPTH);

            }

            write_color(std::cout, pixel_color, SAMPLES_PER_PIXEL);

        }

    }

    std::cerr << "\nDone.\n";

}