//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_MAIN_H
#define RAYTRACER_MAIN_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>
#include <fstream>


//Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

//Utility Functions
inline double degrees_to_radians(double degrees){

    return degrees * pi/180.0;

}

inline double random_double(){

    //Returns a random real in [0,1)
    return rand()/ (RAND_MAX + 1.0);

}

inline double random_double(double min, double max){

    //Returns a random real in [min,max)
    return min + (max-min)*random_double();

}

inline double clamp(double x, double min, double max){

    if(x < min){

        return min;

    }

    if(x > max){

        return max;

    }

    return x;

}

double* multiplyVectorByMatrix(double A[4][4], double v[4]){

     double* u = new double[3];

     for(int i = 0; i < 4; i++){

         u[i] = 0;

         for(int j = 0; j < 4; j++){

            u[i] += A[i][j] * v[j];

         }
     }

    return u;
}

//Common Headers
#include "Ray.h"
#include "Vector3.h"

#endif //RAYTRACER_MAIN_H
