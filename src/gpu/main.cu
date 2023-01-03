//
// Created by moura on 30/12/2022.
//

#include <memory>

#include "../image.h"


#include "util.cuh"
#include "Camera.cuh"
#include "Texture.cuh"
#include "hittable/Hittable.cuh"
#include "hittable/HittableList.cuh"
#include "hittable/Sphere.cuh"
#include "material/Material.cuh"
#include "material/Lambertian.cuh"
#include "material/DiffuseLight.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define RND (curand_uniform(&local_rand_state))

__device__ Color rayColor(const Ray& r, const Color* background, Hittable* world, int depth, curandState* localRandState) {
    Ray currentRay = r;
    Color currentColor = Color(1.0f,1.0f,1.0f);

    for(int d = 0; d < depth; d++) {
        HitRecord rec;
        if(!world->hit(currentRay,0.001,infinity,rec)) return currentColor;

        Ray scattered;
        Color attenuation;
        Color emitted = rec.matPtr->emitted(rec.u, rec.v, rec.p);

        if(!rec.matPtr->scatter(currentRay,rec,attenuation,scattered,localRandState)) return currentColor * emitted;

        currentColor = emitted + attenuation*currentColor;
        currentRay = scattered;
    
    }

    return Color(0.0f,0.0f,0.0f);
    
}

__global__ void renderInit(int maxX, int maxY, curandState* randState){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;


    if((i >= maxX) || (j >= maxY)){

        return;
    };

    int pixel_index = j*maxX + i;

    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &randState[pixel_index]);

}

__global__ void rayTracePixel(Color* frameBuffer, 
                              int imageWidth, 
                              int imageHeight, 
                              int samplesPerPixel, 
                              int maxDepth, 
                              Color** background, 
                              Camera** cam, 
                              Hittable** world, 
                              curandState* randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= imageWidth) || (j >= imageHeight)) return;

    int pixelIndex = j*imageWidth + i;

    curandState localRandState = randState[pixelIndex];

    Color pixelColor(0.0f, 0.0f, 0.0f);

    for(int s = 0; s < samplesPerPixel; s++){
        auto u = (i + randomFloat(&localRandState)) / (imageWidth - 1);
        auto v = (j + randomFloat(&localRandState)) / (imageHeight - 1);

        Ray r = (*cam)->getRay(u, v, &localRandState);
        pixelColor += rayColor(r, *background, *world, maxDepth, &localRandState);

    }

    frameBuffer[pixelIndex] = pixelColor;
}

__global__ void createWorld(Hittable** world, 
                            Hittable** objectList, 
                            int objectListSize,
                            Material** materialList,
                            Texture** textureList,
                            Camera** camera,
                            Color** background, 
                            int imageWidth, 
                            int imageHeight) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        Texture* solidRed = new SolidColor(Color(0.7f, 0.0f, 0.0f));
        Texture* solidGreen = new SolidColor(Color(0.0f, 0.7f, 0.0f));
        Texture* solidBlue = new SolidColor(Color(0.0f, 0.0f, 0.7f));
        Texture* solidGray = new SolidColor(Color(0.7f, 0.7f, 0.7f));
        SolidColor* lightColor = new SolidColor(Color(7.0f,7.0f,7.0f));
        *(textureList+0) = solidRed;
        *(textureList+1) = solidGreen;
        *(textureList+2) = solidBlue;
        *(textureList+3) = solidGray;
        *(textureList+4) = (Texture*)lightColor;

        Material* redLambertian = new Lambertian(solidRed);
        Material* greenLambertian = new Lambertian(solidGreen);
        Material* blueLambertian = new Lambertian(solidBlue);
        Material* grayLambertian = new Lambertian(solidGray);
        Material* diffuseLight = new DiffuseLight(lightColor);
        *(materialList+0) = redLambertian;
        *(materialList+1) = greenLambertian;
        *(materialList+2) = blueLambertian;
        *(materialList+3) = grayLambertian;
        *(materialList+4) = diffuseLight;


        Sphere* redSphere = new Sphere(Point3(0.0f, 20.0f, 0.0f), 10.0f, redLambertian);
        Sphere* greenSphere = new Sphere(Point3(30.0f, 20.0f, 0.0f), 10.0f, greenLambertian);
        Sphere* blueSphere = new Sphere(Point3(60.0f, 20.0f, 0.0f), 10.0f, blueLambertian);
        Sphere* graySphere = new Sphere(Point3(60.0f, -5000.0f, 0.0f), 5009.0f, grayLambertian);
        Sphere* lightSphere1 = new Sphere(Point3(15.0f, 100.0f, 0.0f), 10.0f, diffuseLight);
        Sphere* lightSphere2 = new Sphere(Point3(45.0f, 100.0f, 0.0f), 10.0f, diffuseLight);
        *(objectList+0) = redSphere;
        *(objectList+1) = greenSphere;
        *(objectList+2) = blueSphere;
        *(objectList+3) = graySphere;
        *(objectList+4) = lightSphere1;
        *(objectList+5) = lightSphere2;

        *world = new HittableList(objectList, 4);

        *background = new Color(0.1f,0.1f,0.1f);

        Vector3 lookFrom(30.f,150.0f,-480.0f);
        Vector3 lookAt(30.0f, 15.0f, 10.0f);
        Vector3 viewUp(0.0f,1.0f,0.0f);

        float distToFocus = (lookFrom-lookAt).length();
        float aperture = 0.1f;
        float aspectRatio = float(imageWidth)/float(imageHeight);

        *camera = new Camera(lookFrom,lookAt,viewUp,40.0f,aspectRatio,aperture,distToFocus,0.0f,1.0f);
    } 
}

__global__ void destroyWorld(Hittable** world, 
                            Hittable** objectList, 
                            int objectListSize,
                            Material** materialList,
                            int materialListSize,
                            Texture** textureList,
                            int textureListSize,
                            Camera** camera, 
                            Color** background) {

    if(threadIdx.x == 0 && blockIdx.x == 0) { 
        delete *world;
        delete *camera;
        delete *background;
    
        int i;

        for(i = 0; i < objectListSize; i++) {
            delete objectList[i];
        }

        for(i = 0; i < materialListSize; i++) {
            delete materialList[i];
        }

        for(i = 0; i < textureListSize; i++) {
            delete textureList[i];
        }
    }

}

int main() {
    //Image properties
    const auto ASPECT_RATIO = 1.0f;
    const int IMAGE_WIDTH =  480;
    const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    const int SAMPLES_PER_PIXEL = 400;
    const int MAX_DEPTH = 50;
    const int NUM_PIXELS = IMAGE_WIDTH*IMAGE_HEIGHT;

    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    std::cerr << "Found " << deviceCount << " device(s)" << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;

        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        std::cerr << "Device name: " << prop.name << std::endl;
        std::cerr << "Total Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 << "MB" << std::endl;
        std::cerr << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cerr << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }

    std::cerr << std::endl;

    std::cerr << "Rendering a " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " image ";
    std::cerr << "in " << TILE_WIDTH << "x" << TILE_HEIGHT << " blocks.\n" << std::endl;;


    Color* frameBuffer;
    int frameBufferSize = NUM_PIXELS;
    checkCudaErrors(cudaMallocManaged((void**)&frameBuffer,frameBufferSize*sizeof(Color)));

    curandState* dRandState;
    checkCudaErrors(cudaMalloc((void**)&dRandState, NUM_PIXELS*sizeof(curandState)));

    Color** dBackground;
    checkCudaErrors(cudaMalloc((void**)&dBackground, sizeof(Color*)));

    Camera** dCamera;
    checkCudaErrors(cudaMalloc((void**)&dCamera, sizeof(Camera*)));

    Hittable** dWorld;
    checkCudaErrors(cudaMalloc((void**)&dWorld, sizeof(Hittable*)));

    Hittable** dObjectList;
    int objectListSize = 6;
    checkCudaErrors(cudaMalloc((void**)&dObjectList, objectListSize*sizeof(Hittable*)));

    Material** dMaterialList;
    int materialListSize = 5;
    checkCudaErrors(cudaMalloc((void**)&dMaterialList, materialListSize*sizeof(Material*)));

    Texture** dTextureList;
    int textureListSize = 5;
    checkCudaErrors(cudaMalloc((void**)&dTextureList, textureListSize*sizeof(Texture*)));

    std::cerr << "Creating world..." << std::endl;
    createWorld<<<1,1>>>(dWorld,dObjectList,objectListSize,dMaterialList,dTextureList,dCamera,dBackground,IMAGE_WIDTH,IMAGE_HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "World created!" << std::endl;

    clock_t start, stop;
    start = clock();

    dim3 blocks(IMAGE_WIDTH/TILE_WIDTH+1,IMAGE_HEIGHT/TILE_HEIGHT+1);
    dim3 threads(TILE_WIDTH,TILE_HEIGHT);

    std::cerr << "Initializing rand state..." << std::endl;
    renderInit<<<blocks,threads>>>(IMAGE_WIDTH,IMAGE_HEIGHT,dRandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "Rand state initialized!" << std::endl;

    std::cerr << "Rendering..." << std::endl;
    rayTracePixel<<<blocks,threads>>>(frameBuffer,IMAGE_WIDTH,IMAGE_HEIGHT,SAMPLES_PER_PIXEL,MAX_DEPTH,dBackground,dCamera,dWorld,dRandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();

    std::cerr << "Render finished!" << std::endl;

    double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Render took " << timerSeconds << " seconds.\n";

    std::cerr << "Destroying world..." << std::endl;
    destroyWorld<<<1,1>>>(dWorld,dObjectList,objectListSize,dMaterialList,materialListSize,dTextureList,textureListSize,dCamera,dBackground);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "World destroyed!" << std::endl;



    std::shared_ptr<image::ImageData> imageData = std::make_shared<image::ImageData>();
    imageData->width = IMAGE_WIDTH;
    imageData->height = IMAGE_HEIGHT;
    imageData->componentsPerPixel = BYTES_PER_PIXEL;
    imageData->data = new uint8_t[IMAGE_WIDTH*IMAGE_HEIGHT*BYTES_PER_PIXEL];

    std::cerr << "Writing result..." << std::endl;

    writeFrameBufferToBuffer(imageData->data,frameBuffer,frameBufferSize,SAMPLES_PER_PIXEL);
    //writeBuffer(frameBuffer,IMAGE_WIDTH,IMAGE_HEIGHT);

    std::cerr << "Done!" << std::endl;

    image::writeBMP("result-gpu.bmp",*imageData);

    checkCudaErrors(cudaFree(frameBuffer));
    checkCudaErrors(cudaFree(dRandState));
    checkCudaErrors(cudaFree(dBackground));
    checkCudaErrors(cudaFree(dCamera));
    checkCudaErrors(cudaFree(dWorld));
    checkCudaErrors(cudaFree(dObjectList));
    checkCudaErrors(cudaFree(dMaterialList));
    checkCudaErrors(cudaFree(dTextureList));

    cudaDeviceReset();

}
