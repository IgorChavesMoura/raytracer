#ifndef RAYTRACER_Box_cuh
#define RAYTRACER_Box_cuh
 
#include "Hittable.cuh"
#include "HittableList.cuh"
#include "Rect.cuh"
#include "../util.cuh"
#include "../AABB.cuh"

class Box : public Hittable {
    public:
        __device__ Box() {}
        __device__ Box(const Point3& p0, const Point3& p1, Material* ptr);
        __device__ ~Box() {
            for(int i = 0; i < 6; i++) {
                delete sides.objects[i];
            }

            delete[] sides.objects;
        }

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* randState) const override;

        __device__ virtual bool boundingBox(float t0, float t1, AABB& outputBox) const override;

        __device__ void translate(Vector3 translation);
        __device__ void rotate(float angle);

    public:
        Point3 boxMin;
        Point3 boxMax;
        HittableList sides;
        Transform transform;

};


__device__ Box::Box(const Point3& p0, const Point3& p1, Material* ptr) {
    boxMin = p0;
    boxMax = p1;

    transform.translation.offset = Vector3(0.0f, 0.0f, 0.0f);

    transform.rotation.theta = 0.0f;

    Hittable** sideList = (Hittable**)malloc(6*sizeof(Hittable*)); 

    sideList[0] = new XYRect(boxMin.x(), boxMax.x(), boxMin.y(), boxMax.y(), boxMax.z(), ptr);
    sideList[1] = new XYRect(boxMin.x(), boxMax.x(), boxMin.y(), boxMax.y(), boxMin.z(), ptr);

    sideList[2] = new XZRect(boxMin.x(), boxMax.x(), boxMin.z(), boxMax.z(), boxMax.y(), ptr);
    sideList[3] = new XZRect(boxMin.x(), boxMax.x(), boxMin.z(), boxMax.z(), boxMin.y(), ptr);

    sideList[4] = new YZRect(boxMin.y(), boxMax.y(), boxMin.z(), boxMax.z(), boxMax.x(), ptr);
    sideList[5] = new YZRect(boxMin.y(), boxMax.y(), boxMin.z(), boxMax.z(), boxMin.x(), ptr);

    sides = HittableList(sideList, 6); 

}


__device__ void Box::translate(Vector3 translation) {
    transform.translation.offset = translation;
}

__device__ void Box::rotate(float angle) {
    transform.rotation.theta = degreesToRadians(angle);
    transform.rotation.sinTheta = sin(transform.rotation.theta);
    transform.rotation.cosTheta = cos(transform.rotation.theta);
}

__device__ bool Box::hit(const Ray &r, float tMin, float tMax, HitRecord &rec, curandState* randState) const {
    Ray movedRay(r.origin() - transform.translation.offset, r.direction(), r.time());
    
    if(transform.rotation.theta == 0.0f) {
        bool hit = sides.hit(movedRay, tMin, tMax, rec, randState);
        
        if(hit) {
            rec.p += transform.translation.offset;
        }

        return hit;
    };

    float cosTheta = transform.rotation.cosTheta;
    float sinTheta = transform.rotation.sinTheta;

    auto origin = movedRay.origin();
    auto direction = movedRay.direction();
    
    origin[0] = cosTheta*movedRay.origin()[0] - sinTheta*movedRay.origin()[2];
    origin[2] = sinTheta*movedRay.origin()[0] + cosTheta*movedRay.origin()[2];

    direction[0] = cosTheta*movedRay.direction()[0] - sinTheta*movedRay.direction()[2];
    direction[2] = sinTheta*movedRay.direction()[0] + cosTheta*movedRay.direction()[2];

    Ray rotatedR(origin,direction,movedRay.time());

    if(!sides.hit(rotatedR,tMin,tMax,rec,randState)) return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] = cosTheta*rec.p[0] + sinTheta*rec.p[2];
    p[2] = -sinTheta*rec.p[0] + cosTheta*rec.p[2];

    normal[0] = cosTheta*rec.normal[0] + sinTheta*rec.normal[2];
    normal[2] = -sinTheta*rec.normal[0] + cosTheta*rec.normal[2];

    rec.p = p + transform.translation.offset;
    rec.setFaceNormal(rotatedR, normal);

    return true;
}

__device__ bool Box::boundingBox(float t0, float t1, AABB& outputBox) const {
    AABB bBox = AABB(boxMin, boxMax);
    
    if(transform.rotation.theta == 0.0f) {
        outputBox = AABB(bBox.min() + transform.translation.offset, bBox.max() + transform.translation.offset);
        return true;
    }

    float cosTheta = transform.rotation.cosTheta;
    float sinTheta = transform.rotation.sinTheta;

    float infinity = INFINITY;

    Point3 min(infinity, infinity, infinity);
    Point3 max(-infinity, -infinity, -infinity);

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) {
                auto x = i*bBox.max().x() + (1-i)*bBox.min().x();
                auto y = j*bBox.max().y() + (1-j)*bBox.min().y();
                auto z = k*bBox.max().z() + (1-k)*bBox.min().z();

                auto newx =  cosTheta*x + sinTheta*z;
                auto newz = -sinTheta*x + cosTheta*z;

                Vector3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {

                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bBox = AABB(min + transform.translation.offset,max + transform.translation.offset);

    outputBox = bBox;

    return true;
}

 
#endif // RAYTRACER_Box_cuh