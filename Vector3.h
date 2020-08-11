//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_VECTOR3_H
#define RAYTRACER_VECTOR3_H

#include <cmath>
#include <iostream>

#include "util.h"

class Vector3 {

    public:

        inline static Vector3 random(){

            return Vector3(random_double(), random_double(), random_double());

        }

        inline static  Vector3 random(double t_min, double t_max){

            return Vector3(random_double(t_min, t_max), random_double(t_min, t_max), random_double(t_min, t_max));

        }

        Vector3() : e{0, 0, 0} {}
        Vector3(double e0, double e1, double e2) : e{e0, e1, e2} {}

        double x() const {
            return e[0];
        }

        double y() const {
            return e[1];
        }

        double z() const {
            return e[2];
        }

        double length() const {

            return std::sqrt(length_squared());

        }


        double length_squared() const {

            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];

        }

        void rotate(const Vector3& axis, double angle);

        Vector3 operator-() const {
            return Vector3(-e[0], -e[1], -e[2]);
        }

        double operator[](int i) const {

            return e[i];

        }

        double& operator[](int i){

            return e[i];

        }

        Vector3& operator+=(const Vector3& v){

            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];

            return *this;

        }

        Vector3& operator-=(const Vector3& v){

            e[0] -= v.e[0];
            e[1] -= v.e[1];
            e[2] -= v.e[2];

            return *this;
        }

        Vector3& operator*=(const double t){

            e[0] *= t;
            e[1] *= t;
            e[2] *= t;

            return *this;

        }

        Vector3& operator*=(const Vector3& v){

            e[0] *= v.e[0];
            e[1] *= v.e[1];
            e[2] *= v.e[2];

            return *this;

        }

        Vector3& operator/=(const double t){

            return *this *= 1/t;

        }




    public:
        double e[3];

};


inline std::ostream& operator<<(std::ostream& out, const Vector3& v){

    return out << '[' << v.e[0] << ", " << v.e[1] << ", " << v.e[2] << ']';

}

inline Vector3 operator+(const Vector3& u, const Vector3& v){

    return Vector3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);

}

inline Vector3 operator-(const Vector3& u, const Vector3& v){

    return Vector3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);

}

inline Vector3 operator*(const Vector3& u, const Vector3& v){

    return Vector3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);

}

inline Vector3 operator*(const double t, const Vector3& v){

    return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);

}

inline Vector3 operator*(const Vector3& v, const double t){

    return t * v;

}

inline Vector3 operator/(const Vector3& v, const double t){

    return (1/t) * v;

}


inline double dot(const Vector3& u, const Vector3& v){

    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];

}

inline Vector3 cross(const Vector3& u, const Vector3& v){

    return Vector3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                   u.e[2] * v.e[0] - u.e[0] * v.e[2],
                   u.e[0] * v.e[1] - u.e[1] * v.e[0]);

}

inline Vector3 unit_vector(Vector3 v){

    return v / v.length();

}

void Vector3::rotate(const Vector3 &axis, double angle) {

    double w = std::cos(angle/2);
    double s = std::sin(angle/2);

    Vector3 vPrime = axis*s;

    double x1 = vPrime.x(), y1 = vPrime.y(), z1 = vPrime.z();

    double A[4][4] = {
            { w*w + x1*x1 - y1*y1 - z1*z1, 2*x1*y1 - 2*w*z1, 2*x1*z1 + 2*w*y1, 0 },
            { 2*x1*y1 + 2*w*z1, w*w - x1*x1 + y1*y1 - z1*z1, 2*z1*y1 - 2*w*x1, 0 },
            { 2*x1*z1 - 2*w*y1, 2*z1*y1 + 2*w*z1, w*w - x1*x1 - y1*y1 + z1*z1, 0 },
            { 0, 0, 0, w*w + x1*x1 + y1*y1 + z1*z1 }
    };

    double u[4] = { x(), y(), z(), 1 };

    double* v = multiplyVectorByMatrix(A, u);

    e[0] = v[0];
    e[1] = v[1];
    e[2] = v[2];

}

Vector3 random_in_unit_sphere() {

    while(true) {

        auto p = Vector3::random(-1, 1);

        if(p.length_squared() >= 1){

            continue;

        }

        return p;

    }

}

Vector3 random_unit_vector() {

    auto a = random_double(0, 2*pi);
    auto z = random_double(-1, 1);
    auto r = std::sqrt(1 - z*z);

    return Vector3(r * std::cos(a), r * std::sin(a), z);

}

Vector3 random_in_hemisphere(const Vector3& normal){

    Vector3 in_unit_sphere = random_in_unit_sphere();

    if(dot(in_unit_sphere,normal) > 0.0){

        return in_unit_sphere;

    } else {

        return -in_unit_sphere;

    }

}

Vector3 random_in_unit_disk(){

    while(true){

        auto p = Vector3(random_double(-1, 1), random_double(-1, 1), 0);

        if(p.length_squared() >= 1){

            continue;

        }

        return p;

    }

}

Vector3 reflect(const Vector3& v, const Vector3& n){

    return v - 2*dot(v,n)*n;

}

Vector3 refract(const Vector3& uv, const Vector3& n, double etai_over_etat){

    auto cos_theta = dot(-uv, n);

    Vector3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vector3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;

    return  r_out_perp + r_out_parallel;


}

using point3 = Vector3;
using color = Vector3;

#endif //RAYTRACER_VECTOR3_H
