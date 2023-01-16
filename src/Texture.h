//
// Created by moura on 24/12/2022.
//

#ifndef RAYTRACER_TEXTURE_H
#define RAYTRACER_TEXTURE_H

#include "color.h"
#include "image.h"
#include "Vector3.h"


class Texture {
    public:
        virtual Color value(double u, double v, const Point3& p) const = 0;
};

class SolidColor : public Texture {
    public:
        SolidColor() {}
        SolidColor(Color c) : colorValue(c) {}
        SolidColor(double red, double green, double blue) : SolidColor(Color(red, green, blue)) {}

        virtual Color value(double u, double v, const Vector3& p) const override {
            return colorValue;
        }
    private:
        Color colorValue;
};

class CheckerTexture : public Texture {
    public:
        CheckerTexture() {}
        CheckerTexture(std::shared_ptr<Texture> e, std::shared_ptr<Texture> o) : even(e), odd(o) {}
        CheckerTexture(Color c1, Color c2) : even(std::make_shared<SolidColor>(c1)), odd(std::make_shared<SolidColor>(c2)) {}

    virtual Color value(double u, double v, const Vector3& p) const override {
        auto sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());

        if(sines < 0) {
            return odd->value(u,v,p);
        } else {
            return even->value(u,v,p);
        }
    }

    public:
        std::shared_ptr<Texture> even;
        std::shared_ptr<Texture> odd;
};

class ImageTexture : public Texture {
    public:
        ImageTexture() {}
        ImageTexture(const char* filename) {
            image::readImage(filename, imageData);

            bytesPerScanline = imageData.width * BYTES_PER_PIXEL;
        }

        ~ImageTexture() {
            delete imageData.data;
        }

        virtual Color value(double u, double v, const Vector3& p) const override {
            if(imageData.data == nullptr) return Color(0, 1, 1);

            u = clamp(u, 0.0, 1.0);
            v = 1.0 - clamp(v, 0.0, 1.0); //Flip v to image coordinates

            auto i = static_cast<int>(u*imageData.width);
            auto j = static_cast<int>(v*imageData.height);

            if(i >= imageData.width) i = imageData.width - 1;
            if(j >= imageData.height) j = imageData.height - 1;

            const auto colorScale = 1.0/255.0;
            auto pixel = imageData.data + j*bytesPerScanline + i*BYTES_PER_PIXEL;

            return Color(colorScale * pixel[0], colorScale * pixel[1], colorScale * pixel[2]);
        }

    private:
        image::ImageData imageData;
        int bytesPerScanline;
};

#endif //RAYTRACER_TEXTURE_H
