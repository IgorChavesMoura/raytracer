//
// Created by moura on 30/12/2022.
//

#include <memory>
#include <filesystem>
#include <vector>
#include <cstdio>

#include "../image.h"


#include "util.cuh"
#include "world.cuh"
#include "Camera.cuh"
#include "Texture.cuh"
#include "hittable/Hittable.cuh"
#include "hittable/HittableList.cuh"
#include "hittable/Sphere.cuh"
#include "hittable/Rect.cuh"
#include "material/Material.cuh"
#include "material/Lambertian.cuh"
#include "material/DiffuseLight.cuh"

namespace fs = std::filesystem;

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
    bool hitAnything = false, isBounced = false;

    for(int d = 0; d < depth; d++) {
        HitRecord rec;

        if(!world->hit(currentRay,0.001f,FLT_MAX,rec,localRandState)) break;
        hitAnything = true;

        Ray scattered;
        Color attenuation;
        Color emitted = rec.matPtr->emitted(rec.u,rec.v,rec.p);

        if(!rec.matPtr->scatter(currentRay,rec,attenuation,scattered,localRandState)) {
            if(!isBounced) {
                return emitted;
            }

            return currentColor*emitted;
        }

        currentColor = emitted + attenuation * currentColor;
        currentRay = scattered;
        isBounced = true;
    }

    if(hitAnything) {
        return currentColor*(*background);
    }

    return *background;
    
}

__global__ void printImageData(image::ImageData* imageBuffer, int* imageBufferSize) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for(int i = 0; i < *imageBufferSize; i++) {
            printf("begin imageData %d\n", i);
            image::ImageData imageData = *(imageBuffer + i);
            printf("width: %d, height: %d, cpp: %d\n", imageData.width, imageData.height, imageData.componentsPerPixel);
            size_t dataSize = imageData.width * imageData.height * imageData.componentsPerPixel * sizeof(uint8_t);
            for(long j = 0; j < dataSize; j++) {
                printf("%c\t", imageData.data[j]);
            }

            printf("end imageData %d\n", i);
        }
    }
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

__global__ void createWorld(world::WorldType worldType,
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
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        world::create(worldType,
                      world,
                      objectList,
                      objectListSize,
                      materialList,
                      textureList,
                      imageTextureBuffer,
                      camera,
                      background,
                      imageWidth,
                      imageHeight);
    } 
    
}

__global__ void destroyWorld(Hittable** world, 
                            Hittable** objectList, 
                            int objectListSize,
                            Material** materialList,
                            int materialListSize,
                            Texture** textureList,
                            int textureListSize,
                            image::ImageData* imageTextureBuffer,
                            int* imageTextureBufferSize,
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
    const int IMAGE_WIDTH =  960;
    const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    const int SAMPLES_PER_PIXEL = 10000;
    const int MAX_DEPTH = 50;
    const int NUM_PIXELS = IMAGE_WIDTH*IMAGE_HEIGHT;

    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    std::cerr << "Found " << deviceCount << " device(s)" << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        size_t* deviceStackFrameSize = (size_t*)malloc(sizeof(size_t));
        size_t* deviceMallocHeapSize = (size_t*)malloc(sizeof(size_t));

        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaDeviceGetLimit(deviceStackFrameSize, cudaLimitStackSize));
        checkCudaErrors(cudaDeviceGetLimit(deviceMallocHeapSize, cudaLimitMallocHeapSize));
        std::cerr << "Device name: " << prop.name << std::endl;
        std::cerr << "Total Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 << "MB" << std::endl;
        std::cerr << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cerr << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cerr << "Stack Frame size: " << *deviceStackFrameSize << "B" << std::endl;
        std::cerr << "Malloc Heap size: " << *deviceMallocHeapSize << "B" << std::endl;

        free(deviceStackFrameSize);
        free(deviceMallocHeapSize);
    }   



    std::cerr << std::endl;

    std::cerr << "Rendering a " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " image ";
    std::cerr << "in " << TILE_WIDTH << "x" << TILE_HEIGHT << " blocks.\n" << std::endl;;

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, (size_t)2048L));

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

    image::ImageData* dImageTextureBuffer;
    int* dImageTextureBufferSize;
    //loadImageDataBufferToDevice(dImageTextureBuffer, dImageTextureBufferSize);
    char* assetsDirectory = "assets";

    std::vector<image::ImageData> imageBuffer;

    for(const auto& entry : fs::directory_iterator(assetsDirectory)) {
        image::ImageData assetImageData;
        image::readImage(entry.path().string().c_str(), assetImageData);
        imageBuffer.push_back(assetImageData);
    }

    int assetsAmount = imageBuffer.size();

    checkCudaErrors(cudaMallocManaged((void**)&dImageTextureBuffer, assetsAmount*sizeof(image::ImageData)));
    checkCudaErrors(cudaMallocManaged((void**)&dImageTextureBufferSize, sizeof(int)));

    *dImageTextureBufferSize = assetsAmount;

   /* auto currentHImage = imageBuffer[0]; 
    auto currentDImage = dImageTextureBuffer;
    checkCudaErrors(cudaMemcpy(&(currentDImage->width), &(currentHImage.width), sizeof(int), cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(&(currentDImage->height), &(currentHImage.height), sizeof(int), cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(&(currentDImage->componentsPerPixel), &(currentHImage.componentsPerPixel), sizeof(int), cudaMemcpyDefault));
    size_t size = currentHImage.width * currentHImage.height * currentHImage.componentsPerPixel * sizeof(uint8_t);
    checkCudaErrors(cudaMallocManaged(&(currentDImage->data), size));
    checkCudaErrors(cudaMemcpy(currentDImage->data, currentHImage.data, size, cudaMemcpyDefault)); */

    for(int i = 0; i < assetsAmount; i++) {
        auto currentHImage = imageBuffer[i];
        auto currentDImage = dImageTextureBuffer + i;
        checkCudaErrors(cudaMemcpy(&(currentDImage->width), &(currentHImage.width), sizeof(int), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(&(currentDImage->height), &(currentHImage.height), sizeof(int), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(&(currentDImage->componentsPerPixel), &(currentHImage.componentsPerPixel), sizeof(int), cudaMemcpyDefault));
        size_t size = currentHImage.width * currentHImage.height * currentHImage.componentsPerPixel * sizeof(uint8_t);
        checkCudaErrors(cudaMallocManaged(&(currentDImage->data), size));
        checkCudaErrors(cudaMemcpy(currentDImage->data, currentHImage.data, size, cudaMemcpyDefault));
    }

    /*printImageData<<<1,1>>>(dImageTextureBuffer, dImageTextureBufferSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());*/

    

    world::WorldType worldType = world::cornellBoxWorld;
    world::WorldInfo worldInfo = world::getWorldInfo(worldType);

    Hittable** dObjectList;
    int objectListSize = worldInfo.objectListSize;
    checkCudaErrors(cudaMalloc((void**)&dObjectList, objectListSize*sizeof(Hittable*)));

    Material** dMaterialList;
    int materialListSize = worldInfo.materialListSize;
    checkCudaErrors(cudaMalloc((void**)&dMaterialList, materialListSize*sizeof(Material*)));

    Texture** dTextureList;
    int textureListSize = worldInfo.textureListSize;
    checkCudaErrors(cudaMalloc((void**)&dTextureList, textureListSize*sizeof(Texture*)));

    std::cerr << "Creating world..." << std::endl;
    createWorld<<<1,1>>>(worldType, dWorld,dObjectList,objectListSize,dMaterialList,dTextureList,dImageTextureBuffer,dCamera,dBackground,IMAGE_WIDTH,IMAGE_HEIGHT);
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
    destroyWorld<<<1,1>>>(dWorld,dObjectList,objectListSize,dMaterialList,materialListSize,dTextureList,textureListSize,dImageTextureBuffer,dImageTextureBufferSize,dCamera,dBackground);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "World destroyed!" << std::endl;

    //destroyImageDataBuffer(dImageTextureBuffer, dImageTextureBufferSize);
    for(int i = 0; i < *dImageTextureBufferSize; i++) {
        checkCudaErrors(cudaFree((dImageTextureBuffer + i)->data));
    }

    checkCudaErrors(cudaFree(dImageTextureBuffer));

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
