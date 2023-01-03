//
// Created by moura on 28/12/2022.
//

#ifndef RAYTRACER_CONSTANTMEDIUM_H
#define RAYTRACER_CONSTANTMEDIUM_H

#include "Hittable.h"
#include "../material/Isotropic.h"
#include "../Texture.h"

class ConstantMedium : public Hittable {
    public:
        ConstantMedium(std::shared_ptr<Hittable> b, double d, std::shared_ptr<Texture> a) : boundary(b), negInvDensity(-1/d),
            phaseFunction(std::make_shared<Isotropic>(a)) {}
        ConstantMedium(std::shared_ptr<Hittable> b, double d, Color c) : boundary(b), negInvDensity(-1 / d), phaseFunction(
                std::make_shared<Isotropic>(c)) {}

        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const override;

        virtual bool boundingBox(double t0, double t1, AABB &outputBox) const override {
            return boundary->boundingBox(t0,t1,outputBox);
        }

    public:
        std::shared_ptr<Hittable> boundary;
        std::shared_ptr<Material> phaseFunction;
        double negInvDensity;
};

bool ConstantMedium::hit(const Ray &r, double tMin, double tMax, HitRecord &rec) const {
    // Print occasional samples when debugging. To enable, set enableDebug true.
    const bool enableDebug = false;
    const bool debugging = enableDebug && randomDouble() < 0.00001;

    HitRecord rec1, rec2;

    if(!boundary->hit(r,-infinity,infinity,rec1)) return false;
    if(!boundary->hit(r,rec1.t+0.0001,infinity,rec2)) return false;

    if(rec1.t < tMin) rec1.t = tMin;
    if(rec2.t > tMax) rec2.t = tMax;

    if(rec1.t >= rec2.t) return false;

    if(rec1.t < 0) rec1.t = 0;

    const auto rayLength = r.direction().length();
    const auto distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
    const auto hitDistance = negInvDensity * log(randomDouble());

    if(hitDistance > distanceInsideBoundary) return false;

    rec.t = rec1.t + hitDistance/rayLength;
    rec.p = r.at(rec.t);

    if(debugging) {
        std::cerr << "hitDistance = " <<  hitDistance << '\n'
                  << "rec.t = " <<  rec.t << '\n'
                  << "rec.p = " <<  rec.p << '\n';
    }

    rec.normal = Vector3(1,0,0); // arbitrary
    rec.frontFace = true; // also arbitrary
    rec.matPtr = phaseFunction;

    return true;
}

#endif //RAYTRACER_CONSTANTMEDIUM_H
