//
// Created by moura on 28/12/2022.
//

#ifndef RAYTRACER_CONSTANTMEDIUM_CUH
#define RAYTRACER_CONSTANTMEDIUM_CUH

#include "Hittable.cuh"
#include "../material/Isotropic.cuh"
#include "../Texture.cuh"
#include "../util.cuh"

class ConstantMedium : public Hittable {
    public:
        __device__ ConstantMedium(Hittable* b, float d, Texture* a) : boundary(b), negInvDensity(-1/d),
            phaseFunction(new Isotropic(a)) {}
        __device__ ConstantMedium(Hittable* b, float d, const Color& c) : boundary(b), negInvDensity(-1/d), 
            phaseFunction(new Isotropic(c)) {}
        
        __device__ ~ConstantMedium() {
            delete boundary;
            delete phaseFunction;
        }

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const override;

        __device__ virtual bool boundingBox(float t0, float t1, AABB &outputBox) const override {
            return boundary->boundingBox(t0,t1,outputBox);
        }

    public:
        Hittable* boundary;
        Material* phaseFunction;
        float negInvDensity;
};

__device__ bool ConstantMedium::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const {
    // Print occasional samples when debugging. To enable, set enableDebug true.
    const bool enableDebug = false;
    const bool debugging = enableDebug && randomFloat(randState) < 0.00001f;

    HitRecord rec1, rec2;

    auto infinity = INFINITY;

    if(!boundary->hit(r,-infinity,infinity,rec1,randState)) return false;
    if(!boundary->hit(r,rec1.t+0.0001f,infinity,rec2,randState)) return false;

    if(rec1.t < tMin) rec1.t = tMin;
    if(rec2.t > tMax) rec2.t = tMax;

    if(rec1.t >= rec2.t) return false;

    if(rec1.t < 0) rec1.t = 0;

    const auto rayLength = r.direction().length();
    const auto distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
    const auto hitDistance = negInvDensity * log(randomFloat(randState));

    if(hitDistance > distanceInsideBoundary) return false;

    rec.t = rec1.t + hitDistance/rayLength;
    rec.p = r.at(rec.t);

    rec.normal = Vector3(1,0,0); // arbitrary
    rec.frontFace = true; // also arbitrary
    rec.matPtr = phaseFunction;

    return true;
}

#endif //RAYTRACER_CONSTANTMEDIUM_CUH
