#ifndef RAYTRACER_Texture_cuh
#define RAYTRACER_Texture_cuh
 
#include "Vector3.cuh"

#include "../image.h"

class Texture {
    public:
        __device__ virtual Color value(float u, float v, const Point3& p) const = 0;
};

class SolidColor : public Texture {
    public:
        __device__ SolidColor() {}
        __device__ SolidColor(Color c) : colorValue(c) {}
        __device__ SolidColor(float red, float green, float blue) : SolidColor(Color(red, green, blue)) {}

        __device__ virtual Color value(float u, float v, const Point3& p) const override {
            return colorValue;
        }
    private:
        Color colorValue;
};

class CheckerTexture : public Texture {
    public:
        __device__ CheckerTexture() {}
        __device__ CheckerTexture(Texture* e, Texture* o) : even(e), odd(o) {}
        __device__ CheckerTexture(Color c1, Color c2) : even(new SolidColor(c1)), odd(new SolidColor(c2)) {}

    __device__ virtual Color value(float u, float v, const Point3& p) const override {
        auto sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());

        if(sines < 0) {
            return odd->value(u,v,p);
        } else {
            return even->value(u,v,p);
        }
    }

    public:
        Texture* even;
        Texture* odd;
};

class ImageTexture : public Texture {
    public:
        __device__ ImageTexture() {}
        __device__ ImageTexture(image::ImageData* id) : imageData(id) {
            bytesPerScanline = imageData->width * BYTES_PER_PIXEL;
        }

        __device__ virtual Color value(float u, float v, const Point3& p) const override {
            if(imageData->data == nullptr) return Color(0.0f, 1.0f, 1.0f);

            u = clamp(u, 0.0f, 1.0f);
            v = 1.0 - clamp(v, 0.0f, 1.0f); //Flip v to image coordinates

            auto i = static_cast<int>(u*imageData->width);
            auto j = static_cast<int>(v*imageData->height);

            if(i >= imageData->width) i = imageData->width - 1;
            if(j >= imageData->height) j = imageData->height - 1;

            const auto colorScale = 1.0f/255.0f;
            auto pixel = imageData->data + j*bytesPerScanline + i*BYTES_PER_PIXEL;

            return Color(colorScale * pixel[0], colorScale * pixel[1], colorScale * pixel[2]);
        }

    private:
        image::ImageData* imageData;
        int bytesPerScanline;
};

#endif // RAYTRACER_Texture_cuh