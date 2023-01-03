//
// Created by moura on 24/12/2022.
//

#ifndef RAYTRACER_IMAGE_H
#define RAYTRACER_IMAGE_H
// Disable pedantic warnings for this external library.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../external/stb_image.h"
#include "../external/stb_image_write.h"

// Restore warning levels.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
    #pragma warning (pop)
#endif

#define BYTES_PER_PIXEL 3

namespace image {

    struct ImageData {
        int width, height;
        int componentsPerPixel;
        uint8_t* data;
    };

    void readImage(const char* filename, ImageData& imageData) {
        auto componentsPerPixel = BYTES_PER_PIXEL;

        imageData.data = stbi_load(filename, &(imageData.width), &(imageData.height), &(imageData.componentsPerPixel), componentsPerPixel);
    }

    void writePNG(const char* filename, const ImageData& imageData) {
        stbi_write_png(filename, imageData.width, imageData.height, imageData.componentsPerPixel, imageData.data, imageData.width*imageData.componentsPerPixel);
    }

    void writeBMP(const char* filename, const ImageData& imageData) {
        stbi_write_bmp(filename, imageData.width, imageData.height, imageData.componentsPerPixel, imageData.data);
    }

    void writeJPG(const char* filename, const ImageData& imageData) {
        stbi_write_jpg(filename, imageData.width, imageData.height, imageData.componentsPerPixel, imageData.data, 100);
    }
}

#endif //RAYTRACER_IMAGE_H
