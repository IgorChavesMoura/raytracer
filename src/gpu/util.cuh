//
// Created by moura on 30/12/2022.
//

#ifndef RAYTRACER_UTIL_CUH
#define RAYTRACER_UTIL_CUH

#include <iostream>
#include <cmath>
#include <float.h>

#include <curand_kernel.h>

//Common Headers
#include "Ray.cuh"
#include "Vector3.cuh"

#define CUDART_PI_F 3.141592654f
#define PI 3.141592654f
#define RANDVEC3 Vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

//Constants
const float infinity = FLT_MAX;

//Utility Functions
__device__ inline float degreesToRadians(float degrees){

    return degrees * PI/180.0f;

}

__device__ inline float randomFloat(curandState* randState){

    return curand_uniform(randState);

}

__device__ inline float randomFloat(curandState* randState, float min, float max){

    //Returns a random real in [min,max)
    return min + (max-min) * randomFloat(randState);

}

__device__ inline int randomInt(curandState* randState, int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(randomFloat(randState, min, max + 1));
}

__device__ inline int intmax(int a, int b) { return a > b ? a : b; }

__host__ __device__ inline float clamp(float x, float min, float max){

    if(x < min){

        return min;

    }

    if(x > max){

        return max;

    }

    return x;

}

__device__ float* multiplyVectorByMatrix(float A[4][4], float v[4]){

    float* u = new float[3];

    for(int i = 0; i < 4; i++){

        u[i] = 0;

        for(int j = 0; j < 4; j++){

            u[i] += A[i][j] * v[j];

        }
    }

    return u;
}

__device__ Vector3 Vector3::random(curandState* randState){

    return Vector3(randomFloat(randState), randomFloat(randState), randomFloat(randState));

}

__device__ Vector3 Vector3::random(curandState* randState, float tMin, float tMax){

    return Vector3(randomFloat(randState, tMin, tMax), randomFloat(randState, tMin, tMax), randomFloat(randState, tMin, tMax));

}

__device__ void Vector3::rotate(const Vector3 &axis, float angle) {

    float w = std::cos(angle/2);
    float s = std::sin(angle/2);

    Vector3 vPrime = axis*s;

    float x1 = vPrime.x(), y1 = vPrime.y(), z1 = vPrime.z();

    float A[4][4] = {
            { w*w + x1*x1 - y1*y1 - z1*z1, 2*x1*y1 - 2*w*z1, 2*x1*z1 + 2*w*y1, 0.0f },
            { 2*x1*y1 + 2*w*z1, w*w - x1*x1 + y1*y1 - z1*z1, 2*z1*y1 - 2*w*x1, 0.0f },
            { 2*x1*z1 - 2*w*y1, 2*z1*y1 + 2*w*z1, w*w - x1*x1 - y1*y1 + z1*z1, 0.0f },
            { 0.0f, 0.0f, 0.0f, w*w + x1*x1 + y1*y1 + z1*z1 }
    };

    float u[4] = { x(), y(), z(), 1 };

    float* v = multiplyVectorByMatrix(A, u);

    e[0] = v[0];
    e[1] = v[1];
    e[2] = v[2];

}

__device__ Vector3 randomInUnitSphere(curandState* randState) {

    while(true) {

        auto p = Vector3::random(randState, -1, 1);

        if(p.length_squared() >= 1){

            continue;

        }

        return p;

    }

}

__device__ Vector3 randomUnitVector(curandState* randState) {

    auto a = randomFloat(randState, 0, 2 * PI);
    auto z = randomFloat(randState, -1, 1);
    auto r = std::sqrt(1 - z*z);

    return Vector3(r * std::cos(a), r * std::sin(a), z);

}

__device__ Vector3 randomInHemisphere(curandState* randState, const Vector3& normal){

    Vector3 in_unit_sphere = randomInUnitSphere(randState);

    if(dot(in_unit_sphere,normal) > 0.0f){

        return in_unit_sphere;

    } else {

        return -in_unit_sphere;

    }

}

__device__ Vector3 randomInUnitDisk(curandState* randState){

    while(true){

        auto p = Vector3(randomFloat(randState,-1, 1), randomFloat(randState,-1, 1), 0);

        if(p.length_squared() >= 1){

            continue;

        }

        return p;

    }

}

__host__ void writeFrameBufferToBuffer(unsigned char* buffer, Color* frameBuffer, int frameBufferSize, int samplesPerPixel) {
    int chunk = 0;
    for(int i = frameBufferSize - 1; i >= 0; i--) {
        auto pixelColor = frameBuffer[i];

        auto r = pixelColor.x();
        auto g = pixelColor.y();
        auto b = pixelColor.z();

        // Divide the Color by the number of samples.
        auto scale = 1.0f / samplesPerPixel;
        r = std::sqrt(scale * r);
        g = std::sqrt(scale * g);
        b = std::sqrt(scale * b);

        r = 255 * clamp(r, 0.0f, 0.999f);
        g = 255 * clamp(g, 0.0f, 0.999f);
        b = 255 * clamp(b, 0.0f, 0.999f);

        buffer[chunk + 0] = static_cast<unsigned char>(r);
        buffer[chunk + 1] = static_cast<unsigned char>(g);
        buffer[chunk + 2] = static_cast<unsigned char>(b);

        chunk += 3;
    }
}

__host__ void writeBuffer(Color* frame_buffer, int nx, int ny){

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int i = ny - 1; i >= 0; i--) {
        for (int j = 0; j < nx; j++) {

            size_t pixel_index = i * nx + j;

            float r = frame_buffer[pixel_index].r();
            float g = frame_buffer[pixel_index].g();
            float b = frame_buffer[pixel_index].b();

            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);

            std::cout << ir << " " << ig << " " << ib << "\n";

        }
    }

}





#endif //RAYTRACER_UTIL_CUH
