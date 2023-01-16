#ifndef RAYTRACER_world_cuh
#define RAYTRACER_world_cuh

#include "Camera.cuh"
#include "Texture.cuh"
#include "hittable/Hittable.cuh"
#include "hittable/HittableList.cuh"
#include "hittable/Sphere.cuh"
#include "hittable/Rect.cuh"
#include "hittable/Box.cuh"
#include "hittable/ConstantMedium.cuh"
#include "material/Material.cuh"
#include "material/Lambertian.cuh"
#include "material/DiffuseLight.cuh"
#include "material/Dielectric.cuh"
#include "material/Isotropic.cuh"
#include "material/Metal.cuh"
 
namespace world {

    __host__ __device__ enum WorldType {
        rgbSpheresWorld,
        rgbBoxesWorld,
        cornellBoxWorld
    };

    __host__ struct WorldInfo {
        int objectListSize;
        int materialListSize;
        int textureListSize;
    };

    __host__ WorldInfo getWorldInfo(WorldType worldType) {
        int objectListSize = 0;
        int materialListSize = 0;
        int textureListSize = 0;

        switch(worldType) {
            case rgbSpheresWorld:
                objectListSize = 5;
                materialListSize = 5;
                textureListSize = 5;
                break;
            case rgbBoxesWorld:
                objectListSize = 5;
                materialListSize = 5;
                textureListSize = 5;
                break;
            case cornellBoxWorld:
                objectListSize = 9;
                materialListSize = 7;
                textureListSize = 5;
                break;
            default: 
                break;
        }

        WorldInfo worldInfo;
        worldInfo.objectListSize = objectListSize;
        worldInfo.materialListSize = materialListSize;
        worldInfo.textureListSize = textureListSize;

        return worldInfo;
    }

    __device__ void rgbSpheres(
                            Hittable** world, 
                            Hittable** objectList, 
                            int objectListSize,
                            Material** materialList,
                            Texture** textureList,
                            image::ImageData* imageTextureBuffer,
                            Camera** camera,
                            Color** background, 
                            int imageWidth, 
                            int imageHeight) {
        
        Texture* solidRed = new SolidColor(Color(0.7f, 0.0f, 0.0f));
        Texture* solidGreen = new SolidColor(Color(0.0f, 0.7f, 0.0f));
        Texture* solidBlue = new SolidColor(Color(0.0f, 0.0f, 0.7f));
        Texture* solidGray = new SolidColor(Color(0.7f, 0.7f, 0.7f));
        Texture* lightColor = new SolidColor(Color(7.0f,7.0f,7.0f));
        textureList[0] = solidRed;
        textureList[1]  = solidGreen;
        textureList[2]  = solidBlue;
        textureList[3]  = solidGray;
        textureList[4]  = (Texture*)lightColor;

        Material* redLambertian = new Lambertian(solidRed);
        Material* greenLambertian = new Lambertian(solidGreen);
        Material* blueLambertian = new Lambertian(solidBlue);
        Material* grayLambertian = new Lambertian(solidGray);
        Material* diffuseLight = new DiffuseLight((SolidColor*)lightColor);
        materialList[0] = redLambertian;
        materialList[1] = greenLambertian;
        materialList[2] = blueLambertian;
        materialList[3] = grayLambertian;
        materialList[4] = diffuseLight;



        Sphere* redSphere = new Sphere(Point3(0.0f, 40.0f, 0.0f), 40.0f, redLambertian);
        Sphere* greenSphere = new Sphere(Point3(70.0f, 40.0f, 0.0f), 40.0f, greenLambertian);
        Sphere* blueSphere = new Sphere(Point3(140.0f, 40.0f, 0.0f), 40.0f, blueLambertian);
        Sphere* graySphere = new Sphere(Point3(60.0f, -5000.0f, 0.0f), 5009.0f, grayLambertian);
        YZRect* lightRect = new YZRect(10.0f, 100.0f, -80.0f, 80.0f, 210.0f, diffuseLight);
        objectList[0] = redSphere;
        objectList[1] = greenSphere;
        objectList[2] = blueSphere;
        objectList[3] = graySphere;
        objectList[4] = lightRect;


        *world = new HittableList(objectList, objectListSize);

        *background = new Color(0.0f,0.0f,0.0f);

        Vector3 lookFrom(10.0f,120.0f,-300.0f);
        Vector3 lookAt(95.0f, 45.0f, 10.0f);
        Vector3 viewUp(0.0f,1.0f,0.0f);

        float distToFocus = (lookFrom-lookAt).length();
        float aperture = 0.1f;
        float aspectRatio = float(imageWidth)/float(imageHeight);

        *camera = new Camera(lookFrom,lookAt,viewUp,40.0f,aspectRatio,aperture,distToFocus,0.0f,1.0f);                        
    }

    __device__ void rgbBoxes(
                            Hittable** world, 
                            Hittable** objectList, 
                            int objectListSize,
                            Material** materialList,
                            Texture** textureList,
                            image::ImageData* imageTextureBuffer,
                            Camera** camera,
                            Color** background, 
                            int imageWidth, 
                            int imageHeight) {
        
        Texture* solidRed = new SolidColor(Color(0.7f, 0.0f, 0.0f));
        Texture* solidGreen = new SolidColor(Color(0.0f, 0.7f, 0.0f));
        Texture* solidBlue = new SolidColor(Color(0.0f, 0.0f, 0.7f));
        Texture* solidGray = new SolidColor(Color(0.7f, 0.7f, 0.7f));
        Texture* lightColor = new SolidColor(Color(7.0f,7.0f,7.0f));
        textureList[0] = solidRed;
        textureList[1]  = solidGreen;
        textureList[2]  = solidBlue;
        textureList[3]  = solidGray;
        textureList[4]  = (Texture*)lightColor;

        Material* redLambertian = new Lambertian(solidRed);
        Material* greenLambertian = new Lambertian(solidGreen);
        Material* blueLambertian = new Lambertian(solidBlue);
        Material* grayLambertian = new Lambertian(solidGray);
        Material* diffuseLight = new DiffuseLight((SolidColor*)lightColor);
        materialList[0] = redLambertian;
        materialList[1] = greenLambertian;
        materialList[2] = blueLambertian;
        materialList[3] = grayLambertian;
        materialList[4] = diffuseLight;



        Box* redBox = new Box(Point3(0.0f, 20.0f, 0.0f), Point3(50.0f, 70.0f, 70.0f), redLambertian);
        Box* greenBox = new Box(Point3(70.0f, 20.0f, 0.0f), Point3(120.0f, 70.0f, 70.0f), greenLambertian);
        Box* blueBox = new Box(Point3(140.0f, 20.0f, 0.0f), Point3(190.0f, 70.0f, 70.0f), blueLambertian);
        Sphere* graySphere = new Sphere(Point3(60.0f, -5000.0f, 0.0f), 5009.0f, grayLambertian);
        YZRect* lightRect = new YZRect(10.0f, 100.0f, -80.0f, 80.0f, 210.0f, diffuseLight);
        objectList[0] = redBox;
        objectList[1] = greenBox;
        objectList[2] = blueBox;
        objectList[3] = graySphere;
        objectList[4] = lightRect;


        *world = new HittableList(objectList, objectListSize);

        *background = new Color(0.0f,0.0f,0.0f);

        Vector3 lookFrom(10.0f,120.0f,-300.0f);
        Vector3 lookAt(95.0f, 45.0f, 10.0f);
        Vector3 viewUp(0.0f,1.0f,0.0f);

        float distToFocus = (lookFrom-lookAt).length();
        float aperture = 0.1f;
        float aspectRatio = float(imageWidth)/float(imageHeight);

        *camera = new Camera(lookFrom,lookAt,viewUp,40.0f,aspectRatio,aperture,distToFocus,0.0f,1.0f);                        
    }

    __device__ void cornellBox(Hittable** world, 
                    Hittable** objectList, 
                    int objectListSize,
                    Material** materialList,
                    Texture** textureList,
                    image::ImageData* imageTextureBuffer,
                    Camera** camera,
                    Color** background, 
                    int imageWidth, 
                    int imageHeight) {

        Texture* redColor = new SolidColor(0.65f, 0.05f, 0.05f);
        Texture* whiteColor = new SolidColor(0.73f, 0.73f, 0.73f);
        Texture* greenColor = new SolidColor(0.12f, 0.45f, 0.15f);
        Texture* lightColor = new SolidColor(7.0f, 7.0f, 7.0f);
        image::ImageData* earthImage = imageTextureBuffer;
        Texture* earthTexture = new ImageTexture(imageTextureBuffer);
        textureList[0] = redColor;
        textureList[1]  = whiteColor;
        textureList[2]  = greenColor;
        textureList[3]  = lightColor;
        textureList[4] = earthTexture;
        
        Material* redLambertian = new Lambertian(redColor);
        Material* whiteLambertian = new Lambertian(whiteColor);
        Material* greenLambertian = new Lambertian(greenColor);
        Material* dielectric = new Dielectric(1.5f);
        Material* metal = new Metal(Color(0.73f, 0.73f, 0.73f), 0.0f);
        //Material* fogMaterial = new Isotropic(whiteColor);
        Material* diffuseLight = new DiffuseLight((SolidColor*)lightColor);
        Material* earthMaterial = new Lambertian(earthTexture);
        materialList[0] = redLambertian;
        materialList[1] = greenLambertian;
        materialList[2] = whiteLambertian;
        materialList[3] = diffuseLight;
        materialList[4] = dielectric;
        materialList[5] = metal;
        materialList[6] = earthMaterial;

        Hittable* greenWall = new YZRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, greenLambertian);
        Hittable* redWall = new YZRect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, redLambertian);
        Hittable* whiteWall = new XYRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, whiteLambertian);
        Hittable* ceiling = new XZRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, whiteLambertian);
        Hittable* floor = new XZRect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, whiteLambertian); 
        Hittable* light = new XZRect(113.0f, 443.0f, 127.0f, 432.0f, 554.0f, diffuseLight);
        Hittable* box1 = new Box(Point3(0.0f, 0.0f, 0.0f), Point3(165.0f, 330.0f, 165.0f), metal);
        Hittable* box2 = new Box(Point3(0.0f, 0.0f, 0.0f), Point3(165.0f, 165.0f, 165.0f), dielectric);
        ((Box*)box1)->rotate(15.0f);
        ((Box*)box1)->translate(Vector3(265.0f,0.0f,295.0f));
        ((Box*)box2)->rotate(-18.0f);
        ((Box*)box2)->translate(Vector3(130.0f,0.0f,65.0f));
        Hittable* earth = new Sphere(Point3(192.5f, 280.0f, 210.0f), 100.0f, earthMaterial);
        //Hittable* box1Fog = new ConstantMedium(box1, 0.007f, Color(1.0f,1.0f,1.0f));
        //Hittable* box2 = new Box(Point3(0.0f, 0.0f, 0.0f), Point3(165.0f, 165.0f, 165.0f), whiteLambertian);
        objectList[0] = greenWall;
        objectList[1] = redWall;
        objectList[2] = whiteWall;
        objectList[3] = ceiling;
        objectList[4] = floor;
        objectList[5] = light;
        objectList[6] = box1;
        objectList[7] = box2;
        objectList[8] = earth;

        *world = new HittableList(objectList, objectListSize);
        *background = new Color(0.0f,0.0f,0.0f);

        Point3 lookFrom(278.0f, 278.0f, -800.0f);
        Point3 lookAt(278.0f, 278.0f, 0.0f);
        Vector3 viewUp(0.0f,1.0f,0.0f);
        float distToFocus = 10.0f;
        float aperture = 0.0f;
        float aspectRatio = float(imageWidth)/float(imageHeight);
        *camera = new Camera(lookFrom,lookAt,viewUp,40.0f,aspectRatio,aperture,distToFocus,0.0f,1.0f);
        
    }

    __device__ void create(WorldType worldType,
                                Hittable** world, 
                                Hittable** objectList, 
                                int objectListSize,
                                Material** materialList,
                                Texture** textureList,
                                image::ImageData* imageTextureBuffer,
                                Camera** camera,
                                Color** background, 
                                int imageWidth, 
                                int imageHeight) {
        switch(worldType) {
            case rgbSpheresWorld:
                rgbSpheres(world,objectList,objectListSize,materialList,textureList,imageTextureBuffer,camera,background,imageWidth,imageHeight);
                break;
            case rgbBoxesWorld:
                rgbBoxes(world,objectList,objectListSize,materialList,textureList,imageTextureBuffer,camera,background,imageWidth,imageHeight);
                break;
            case cornellBoxWorld:
                cornellBox(world,objectList,objectListSize,materialList,textureList,imageTextureBuffer,camera,background,imageWidth,imageHeight);
                break;
            default: 
                break;
        }
    } 

}
 
#endif // RAYTRACER_world_cuh