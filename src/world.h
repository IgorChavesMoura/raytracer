//
// Created by moura on 26/12/2022.
//

#ifndef RAYTRACER_WORLD_H
#define RAYTRACER_WORLD_H

#include "hittable/HittableList.h"
#include "hittable/Sphere.h"
#include "hittable/ConstantMedium.h"
#include "hittable/MovingSphere.h"
#include "hittable/Rect.h"
#include "hittable/Box.h"
#include "hittable/Translate.h"
#include "hittable/Rotate.h"
#include "hittable/FlipNormals.h"
#include "material/Lambertian.h"
#include "material/Metal.h"
#include "material/Dielectric.h"
#include "material/DiffuseLight.h"
#include "Texture.h"
#include "BVHNode.h"

namespace worlds {

    void testLight(HittableList& world, Point3& lookFrom, Point3& lookAt, Vector3& vup, double& distToFocus, double& aperture) {
        auto solidRed = Color(0.7, 0.0, 0.0);
        auto solidGreen = Color(0.0, 0.7, 0.0);
        auto solidBlue = Color(0.0, 0.0, 7.0);
        auto solidGray = Color(0.7, 0.7, 0.7);
        auto lightColor = Color(7.0, 7.0, 7.0);

        auto redLambertian = std::make_shared<Lambertian>(solidRed);
        auto greenLambertian = std::make_shared<Lambertian>(solidGreen);
        auto blueLambertian = std::make_shared<Lambertian>(solidBlue);
        auto grayLambertian = std::make_shared<Lambertian>(solidGray);
        auto diffuseLight = std::make_shared<DiffuseLight>(lightColor);

        auto redSphere = std::make_shared<Sphere>(Point3(0.0, 20.0, 0.0), 10.0, redLambertian);
        auto greenSphere = std::make_shared<Sphere>(Point3(30.0, 20.0, 0.0), 10.0, greenLambertian);
        auto blueSphere = std::make_shared<Sphere>(Point3(60.0, 20.0, 0.0), 10.0, blueLambertian);
        auto graySphere = std::make_shared<Sphere>(Point3(60.0, -5000.0, 0.0), 5009.0, grayLambertian);
        auto lightSphere1 = std::make_shared<Sphere>(Point3(15.0, 100.0, 0.0), 10.0, diffuseLight);
        auto lightSphere2 = std::make_shared<Sphere>(Point3(45.0, 100.0, 0.0), 10.0, diffuseLight);

        world.add(redSphere);
        world.add(greenSphere);
        world.add(blueSphere);
        world.add(graySphere);
        world.add(lightSphere1);
        world.add(lightSphere2);

        lookFrom = Vector3(30.,150.0,-480.0);
        lookAt = Vector3(30.0, 15.0, 10.0);
        vup = Vector3(0.0, 1.0, 0.0);

        distToFocus = (lookFrom - lookAt).length();
        aperture = 0.1;

    }

    void movingSpheres(HittableList& world, Point3& lookFrom, Point3& lookAt, Vector3& vup, double& distToFocus, double& aperture) {
        auto checker = std::make_shared<CheckerTexture>(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));
        auto material_ground = std::make_shared<Lambertian>(checker);
        world.add(std::make_shared<Sphere>(Point3(0, -1000.5, -1), 1000, material_ground));

        auto albedo = Color::random() * Color::random();

        std::shared_ptr<Material> mat = std::make_shared<Lambertian>(albedo);

        for(int a = -11; a < 11; a++){

            for(int b = -11; b < 11; b++){

                auto choose_mat = randomDouble();

                Point3 center(a + 0.9 * randomDouble(), 0.2, b + 0.9 * randomDouble());

                if((center - Point3(4, 0.2, 0)).length() > 0.9){

                    std::shared_ptr<Material> sphere_material;

                    if(choose_mat < 0.8){

                        auto albedo = Color::random() * Color::random();

                        sphere_material = std::make_shared<Lambertian>(albedo);

                        auto center2 = center + Vector3(0, randomDouble(0, 0.5), 0);

                        world.add(std::make_shared<MovingSphere>(center, center2, 0.0, 1.0, 0.2, sphere_material));

                    } else if(choose_mat < 0.95){

                        auto albedo = Color::random(0.5, 1);
                        auto fuzz = randomDouble(0, 0.5);

                        sphere_material = std::make_shared<Metal>(albedo, fuzz);
                        world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));

                    } else {

                        sphere_material = std::make_shared<Dielectric>(1.5);

                        world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));

                    }

                }


            }

        }

        lookFrom = Point3(13, 2, 3);
        lookAt = Point3(0, 0, 0);
        vup = Vector3(0,1,0);

        distToFocus = (lookFrom - lookAt).length();
        aperture = 0.1;
    }

    void simpleLight(HittableList& world, Point3& lookFrom, Point3& lookAt, Vector3& vup, double& distToFocus, double& aperture) {
        auto blueColorTexture = std::make_shared<SolidColor>(Color(0.0, 0.0, 1.0));
        auto redColorTexture = std::make_shared<SolidColor>(Color(1.0, 0.0, 0.0));

        world.add(std::make_shared<Sphere>(Point3(0, -1000, 0), 1000, std::make_shared<Lambertian>(blueColorTexture)));
        world.add(std::make_shared<Sphere>(Point3(0, 2, 0), 2, std::make_shared<Lambertian>(redColorTexture)));

        auto diffLight = std::make_shared<DiffuseLight>(Color(4, 4, 4));

        world.add(std::make_shared<XYRect>(3,5,1,3,-2,diffLight));


        lookFrom = Point3(26, 3, 6);
        lookAt = Point3(0, 2, 0);
        vup = Vector3(0,1,0);

        distToFocus = (lookFrom - lookAt).length();
        aperture = 0.1;
    }

    void cornellBox(HittableList& world, Point3& lookFrom, Point3& lookAt, Vector3& vup, double& distToFocus, double& aperture) {
        auto red   = std::make_shared<Lambertian>(Color(.65, .05, .05));
        auto white = std::make_shared<Lambertian>(Color(.73, .73, .73));
        auto green = std::make_shared<Lambertian>(Color(.12, .45, .15));
        auto light = std::make_shared<DiffuseLight>(Color(7, 7, 7));

        world.add(std::make_shared<YZRect>(0, 555, 0, 555, 555, green));
        world.add(std::make_shared<YZRect>(0, 555, 0, 555, 0, red));
        world.add(std::make_shared<XZRect>(113, 443, 127, 432, 554, light));
        world.add(std::make_shared<XZRect>(0, 555, 0, 555, 0, white));
        world.add(std::make_shared<XZRect>(0, 555, 0, 555, 555, white));
        world.add(std::make_shared<XYRect>(0, 555, 0, 555, 555, white));

        std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Point3(0, 0, 0), Point3(165, 330, 165), white);
        box1 = std::make_shared<RotateY>(box1,15);
        box1 = std::make_shared<Translate>(box1,Vector3(265,0,295));
        world.add(box1);

        std::shared_ptr<Hittable> box2 = std::make_shared<Box>(Point3(0,0,0), Point3(165,165,165), white);
        box2 = std::make_shared<RotateY>(box2, -18);
        box2 = std::make_shared<Translate>(box2, Vector3(130,0,65));
        world.add(box2);

        lookFrom = Point3(278, 278, -800);
        lookAt = Point3(278, 278, 0);
        vup = Vector3(0,1,0);

        distToFocus = 10.0;
        aperture = 0.0;
    }

    void cornellBoxEarth(HittableList& world, Point3& lookFrom, Point3& lookAt, Vector3& vup, double& distToFocus, double& aperture) {
        auto red   = std::make_shared<Lambertian>(Color(.65, .05, .05));
        auto white = std::make_shared<Metal>(Color(.73, .73, .73), 1.0);
        auto green = std::make_shared<Lambertian>(Color(.12, .45, .15));
        auto light = std::make_shared<DiffuseLight>(Color(7, 7, 7));

        world.add(std::make_shared<YZRect>(0, 555, 0, 555, 555, green));
        world.add(std::make_shared<YZRect>(0, 555, 0, 555, 0, red));
        world.add(std::make_shared<XZRect>(113, 443, 127, 432, 554, light));
        world.add(std::make_shared<XZRect>(0, 555, 0, 555, 0, white));
        world.add(std::make_shared<XZRect>(0, 555, 0, 555, 555, white));
        world.add(std::make_shared<XYRect>(0, 555, 0, 555, 555, white));

        std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Point3(0, 0, 0), Point3(165, 330, 165), white);
        box1 = std::make_shared<RotateY>(box1,15);
        box1 = std::make_shared<Translate>(box1,Vector3(265,0,295));
        // world.add(box1);

        auto earthTexture = std::make_shared<ImageTexture>("earthmap.jpg");
        auto earthSurface = std::make_shared<Lambertian>(earthTexture);
        auto earth = std::make_shared<Sphere>(Point3(0, 0, 0), 400, earthSurface);
        world.add(std::make_shared<Translate>(earth,Vector3(265,0,295)));

        lookFrom = Point3(278, 278, -800);
        lookAt = Point3(278, 278, 0);
        vup = Vector3(0,1,0);

        distToFocus = 10.0;
        aperture = 0.0;
    }

    void final(HittableList& world, Point3& lookFrom, Point3& lookAt, Vector3& vup, double& distToFocus, double& aperture) {
        HittableList boxes1;
        auto ground = std::make_shared<Lambertian>(Color(0.48, 0.83, 0.53));

        const int boxesPerSide = 20;
        for (int i = 0; i < boxesPerSide; i++) {
            for (int j = 0; j < boxesPerSide; j++) {
                auto w = 100.0;
                auto x0 = -1000.0 + i*w;
                auto z0 = -1000.0 + j*w;
                auto y0 = 0.0;
                auto x1 = x0 + w;
                auto y1 = randomDouble(1, 101);
                auto z1 = z0 + w;

                boxes1.add(std::make_shared<Box>(Point3(x0, y0, z0), Point3(x1, y1, z1), ground));
            }
        }

        world.add(std::make_shared<BVHNode>(boxes1,0,1));

        auto light = std::make_shared<DiffuseLight>(Color(7, 7, 7));
        world.add(std::make_shared<XZRect>(123, 423, 147, 412, 554, light));

        auto boundary = std::make_shared<Sphere>(Point3(360, 150, 145), 70, std::make_shared<Lambertian>(Color(0.0, 0.0, 0.6)));
//        world.add(boundary);
//        world.add(std::make_shared<ConstantMedium>(boundary, 0.2, Color(0.2, 0.4, 0.9)));
        boundary = std::make_shared<Sphere>(Point3(0, 0, 0), 5000, std::make_shared<Dielectric>(1.5));
        world.add(std::make_shared<ConstantMedium>(boundary, .0001, Color(1, 1, 1)));

        auto emat = std::make_shared<Lambertian>(std::make_shared<ImageTexture>("earthmap.jpg"));
        world.add(std::make_shared<Sphere>(Point3(400, 200, 400), 100, emat));

        world.add(std::make_shared<Sphere>(Point3(260, 150, 45), 50, std::make_shared<Dielectric>(1.5)));
        world.add(std::make_shared<Sphere>(
                Point3(0, 150, 145), 50, std::make_shared<Metal>(Color(0.8, 0.8, 0.9), 1.0)
        ));


        lookFrom = Point3(478, 278, -600);
        lookAt = Point3(278, 278, 0);
        vup = Vector3(0,1,0);

        distToFocus = 10.0;
        aperture = 0.0;

    }
}

#endif //RAYTRACER_WORLD_H
