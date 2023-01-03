//
// Created by igor on 08/08/2020.
//

#ifndef RAYTRACER_MESH_H
#define RAYTRACER_MESH_H

#include "Hittable.h"
#include "Triangle.h"
#include "../util.h"

class Mesh : public Hittable {

    public:

        Mesh() { }
        Mesh(std::shared_ptr<Material> m) : material(m) {}

        virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const;

        void add_face(Vector3 v0, Vector3 v1, Vector3 v2){

            faces.push_back(std::make_shared<Triangle>(v0,v1,v2,material));

        }

        void translate(const Vector3& v){

            for(std::shared_ptr<Triangle> f : faces){

                f->v0 += v;
                f->v1 += v;
                f->v2 += v;

            }

        }

        void translateToPos(const Vector3& v){

            for(std::shared_ptr<Triangle> f : faces){

                f->v0 += (v - f->v0);
                f->v1 += (v - f->v1);
                f->v2 += (v - f->v2);

            }

        }

        void scale(const Vector3& v){

            for(std::shared_ptr<Triangle> f : faces){

                f->v0 *= v;
                f->v1 *= v;
                f->v2 *= v;

            }

        }

        void rotate(const Vector3& axis, double angle){

            for(std::shared_ptr<Triangle> f : faces){

                f->v0.rotate(axis,angle);
                f->v1.rotate(axis,angle);
                f->v2.rotate(axis,angle);

            }

        }

    public:
        std::shared_ptr<Material> material;

        std::vector<std::shared_ptr<Triangle>> faces;

};

bool Mesh::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {

    Vector3 normal;

    double tHit = infinity;

    bool hit = false;


    //Check if has hit on any face and get closest hit
    for(std::shared_ptr<Triangle> f : faces){

        if(f->hit(r, tMin, tMax, rec)){

            if(rec.t < tHit){

                tHit = rec.t;
                normal = rec.normal;

            }

            hit = true;

        }

    }

    rec.t = tHit;
    rec.normal = normal;
    rec.p = r.at(tHit);

    return hit;

}


#endif //RAYTRACER_MESH_H
