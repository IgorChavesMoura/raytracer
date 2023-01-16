#ifndef RAYTRACER_HittableList_cuh
#define RAYTRACER_HittableList_cuh
 
#include "Hittable.cuh"

class HittableList : public Hittable {

    public:

        __device__ HittableList() {}
        __device__ HittableList(Hittable** o, int ols) : objects(o), objectListSize(ols) {}

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const override;
        __device__ virtual bool boundingBox(float t0, float t1, AABB& outputBox) const override;

    public:
        Hittable** objects;
        int objectListSize;
};

__device__ bool HittableList::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const {

    HitRecord tempRec;
    bool hitAnything = false;
    auto closestSoFar = tMax;

    for(int i = 0; i < objectListSize; i++){
        auto object = objects[i];
        
        if(object->hit(r, tMin, closestSoFar, tempRec, randState)){

            hitAnything = true;
            closestSoFar = tempRec.t;
            rec = tempRec;

        }
    }

    return hitAnything;

}

__device__ bool HittableList::boundingBox(float t0, float t1, AABB &outputBox) const {
    if(objectListSize == 0) return false;

    AABB tempBox;
    bool firstBox = true;

    for(int i = 0; i < objectListSize; i++) {
        Hittable* object = objects[i];

        if(!object->boundingBox(t0,t1,tempBox)) return false;

        outputBox = firstBox ? tempBox : sorroundingBox(outputBox, tempBox);
        firstBox = false;
    }

    return true;
}
 
#endif // RAYTRACER_HittableList_cuh