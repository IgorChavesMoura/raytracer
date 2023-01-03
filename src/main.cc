#include <iostream>

#include "util.h"
#include "color.h"
#include "world.h"
#include "image.h"
#include "BVHNode.h"
#include "Camera.h"

#define MT 1

Color rayColor(const Ray& r, const Color& background, const Hittable& world, int depth, int maxDepth){

    HitRecord rec;

    if(depth >= maxDepth){

        return Color(0, 0, 0);

    }

    if(!world.hit(r,0.001,infinity,rec)) return background;

    Ray scattered;
    Color attenuation;
    Color emitted = rec.matPtr->emitted(rec.u, rec.v, rec.p);

    if(!rec.matPtr->scatter(r, rec, attenuation, scattered)){

        return emitted;

    }

    return emitted +
           attenuation * rayColor(scattered, background, world, depth + 1, maxDepth);


}

void rayTrace(std::shared_ptr<int> nextTileX,
              std::shared_ptr<int> nextTileY,
              int tileWidth,
              int tileHeight,
              std::shared_ptr<std::mutex> tileInfoMutex,
              const int imageWidth,
              const int imageHeight,
              const int samplesPerPixel,
              const int maxDepth,
              const Color& background,
              const Camera& cam,
              const Hittable& world,
              std::shared_ptr<image::ImageData> imageData) {

    // Run until all tiles have been processed
    while(true) {

        // Use mutex to access shared information about next tile to render
        tileInfoMutex->lock();

        // Get next tile screen position from shared variables
        int tileX = *nextTileX;
        int tileY = *nextTileY;

        // Advance to the next tile
        *nextTileX += tileWidth;

        // Reached end of width
        if(*nextTileX >= imageWidth) {

            // Advance to the next line
            *nextTileX = 0;
            *nextTileY += tileHeight;
        }

        tileInfoMutex->unlock();

        // Terminate when go beyond image borders
        if(tileY >= imageHeight) return;

        // Clip tile to image borders
        tileWidth = tileWidth - intmax(0, tileX + tileWidth - imageWidth);
        tileHeight = tileHeight - intmax(0, tileY + tileHeight - imageHeight);

        for(int j = tileY; j < tileY + tileHeight; j++){
            for(int i = tileX; i < tileX + tileWidth; i++){
                Color pixel_color(0, 0, 0);

                for(int s = 0; s < samplesPerPixel; s++){
                    auto u = (i + randomDouble()) / (imageWidth - 1);
                    auto v = (j + randomDouble()) / (imageHeight - 1);

                    Ray r = cam.getRay(u, v);
                    pixel_color += rayColor(r, background, world, 0, maxDepth);

                }

                int byteIndex = (i + ((imageHeight - 1) - j) * imageWidth) * 3;
                writeColorToBuffer(imageData->data, byteIndex, pixel_color, samplesPerPixel);

            }


        }

    }

}

int main(int argc, char** argv){

    //Image
    const auto ASPECT_RATIO = 1.0;
    const int IMAGE_WIDTH =  800;
    const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
    const int SAMPLES_PER_PIXEL = 10000;
    const int MAX_DEPTH = 50;

    //World
    HittableList world;

    Color background(0, 0, 0);

    //Camera
    Point3 lookfrom;
    Point3 lookat;
    Vector3 vup;

    double dist_to_focus;
    double aperture;

    worlds::final(world,lookfrom,lookat,vup,dist_to_focus,aperture);
    background = Color(0, 0, 0);

    Camera cam(lookfrom, lookat, vup, 40.0, ASPECT_RATIO, aperture, dist_to_focus, 0.0, 1.0);

    BVHNode worldBVHTree = BVHNode(world, 0.0, 1.0);

    std::shared_ptr<image::ImageData> imageData = std::make_shared<image::ImageData>();
    imageData->width = IMAGE_WIDTH;
    imageData->height = IMAGE_HEIGHT;
    imageData->componentsPerPixel = BYTES_PER_PIXEL;
    imageData->data = new uint8_t[IMAGE_WIDTH*IMAGE_HEIGHT*BYTES_PER_PIXEL];

    //Render
    // std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";

#if MT
    const int TILE_WIDTH = 32;
    const int TILE_HEIGHT = 32;

    std::shared_ptr<int> nextTileX = std::make_shared<int>(0);
    std::shared_ptr<int> nextTileY = std::make_shared<int>(0);
    const unsigned int threadsNum = intmax(1, std::thread::hardware_concurrency());

    std::cout << "Number of threads available: " << threadsNum << std::endl;

    std::thread* threads = new std::thread[threadsNum];

    std::cout << "Using " << threadsNum << " threads and " << TILE_WIDTH << "x" << TILE_HEIGHT << " tiles" << std::endl;

    std::shared_ptr<std::mutex> tileInfoMutex = std::make_shared<std::mutex>();

    // Start measuring time
    std::chrono::high_resolution_clock::time_point tStart = std::chrono::high_resolution_clock::now();

    // Spawn threads
    for (unsigned int t = 0; t < threadsNum; t++) {
        threads[t] = std::thread(rayTrace,
                                 nextTileX,
                                 nextTileY,
                                 TILE_WIDTH,
                                 TILE_HEIGHT,
                                 tileInfoMutex,
                                 IMAGE_WIDTH,
                                 IMAGE_HEIGHT,
                                 SAMPLES_PER_PIXEL,
                                 MAX_DEPTH,
                                 std::ref(background),
                                 std::ref(cam),
                                 std::ref(worldBVHTree),
                                 imageData);
    }

    // Wait for threads to finish
    for (unsigned int t = 0; t < threadsNum; t++) {
        threads[t].join();
    }

    // Stop measuring time
    std::chrono::high_resolution_clock::time_point tStop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop - tStart).count();

    std::cout << "Raytracing using " << threadsNum << " threads and " << TILE_WIDTH << "x" << TILE_HEIGHT << " tiles took " << duration << " milliseconds." << std::endl;

#else
    int byteIndex = 0;

    for(int i = (IMAGE_HEIGHT - 1); i >= 0; i--){
        // std::cerr << "\rScanline: " << (i + 1) << " of " << IMAGE_HEIGHT << std::endl << std::flush;
        for(int j = 0; j < IMAGE_WIDTH; j++){ 

            Color pixel_color(0,0,0);

            for(int s = 0; s < SAMPLES_PER_PIXEL; s++){

                auto u = (j + random_double()) / (IMAGE_WIDTH - 1);
                auto v = (i + random_double()) / (IMAGE_HEIGHT - 1);

                Ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, background, worldBVHTree, MAX_DEPTH);

            }

            write_color_to_buffer(imageData->data, byteIndex, pixel_color, SAMPLES_PER_PIXEL);

            byteIndex += 3;
        }


    }

#endif
    image::writeBMP("result.bmp", *imageData);

    delete[] imageData->data;
    std::cout << "\nDone.\n";

}