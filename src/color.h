//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_COLOR_H
#define RAYTRACER_COLOR_H

#include <iostream>

#include "util.h"


void writeColor(std::ostream &out, Color pixel_color, int samples_per_pixel) {

    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the Color by the number of samples.
    auto scale = 1.0/samples_per_pixel;
    r = std::sqrt(scale * r);
    g = std::sqrt(scale * g);
    b = std::sqrt(scale * b);



    // Write the translated [0,255] value of each Color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

void writeColorToBuffer(unsigned char* buffer, int chunk, Color pixelColor, int samplesPerPixel) {
    auto r = pixelColor.x();
    auto g = pixelColor.y();
    auto b = pixelColor.z();

    // Divide the Color by the number of samples.
    auto scale = 1.0 / samplesPerPixel;
    r = std::sqrt(scale * r);
    g = std::sqrt(scale * g);
    b = std::sqrt(scale * b);

    r = 255 * clamp(r, 0.0, 0.999);
    g = 255 * clamp(g, 0.0, 0.999);
    b = 255 * clamp(b, 0.0, 0.999);

    buffer[chunk + 0] = static_cast<unsigned char>(r);
    buffer[chunk + 1] = static_cast<unsigned char>(g);
    buffer[chunk + 2] = static_cast<unsigned char>(b);
}

#endif //RAYTRACER_COLOR_H
