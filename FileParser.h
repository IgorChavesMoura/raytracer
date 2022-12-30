//
// Created by igor on 09/08/2020.
//

#ifndef RAYTRACER_FILEPARSER_H
#define RAYTRACER_FILEPARSER_H

#include "hittable/Mesh.h"
#include "material/Material.h"
#include "material/Lambertian.h"
#include "util.h"

class FileParser {

    public:
        static std::shared_ptr<Mesh> parseStlFile(std::string file_path);

    private:
        static Vector3 parseVector3FromFacet(char* facet);
};

Vector3 FileParser::parseVector3FromFacet(char* facet){

    char f1[4] = { facet[0], facet[1],facet[2],facet[3] };

    char f2[4] = { facet[4], facet[5],facet[6],facet[7] };

    char f3[4] = { facet[8], facet[9],facet[10],facet[11] };

    float xx = *((float*) f1 );
    float yy = *((float*) f2 );
    float zz = *((float*) f3 );

    double x = double(xx);
    double y = double(yy);
    double z = double(zz);

    return Vector3(x, y, z);

}

std::shared_ptr<Mesh> FileParser::parseStlFile(std::string file_path) {

    auto albedo = Color::random() * Color::random();

    std::shared_ptr<Material> mat = std::make_shared<Lambertian>(albedo);

    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(mat);

    std::ifstream file(file_path.c_str(), std::ios::in | std::ios::binary);

    char header_info[80] = "";
    char nTri[4];
    unsigned long nTriLong;

    //Read 80-byte header
    if(file){

        file.read(header_info,80);

        std::cerr << "Header: " << header_info << std::endl;

    } else {

        std::cerr << "Error" << std::endl;

    }

    //Read 4-byte long
    if(file){

        file.read(nTri, 4);

        nTriLong = *((unsigned long*)nTri);

        std::cerr << "n Tri: " << nTriLong << std::endl;

    } else {

        std::cerr << "Error" << std::endl;

    }

    for(int i = 0; i < nTriLong; i++){

        char facet[50];



        if(file){

            //Read one 50-byte triangle
            file.read(facet, 50);


            Vector3 v0 = parseVector3FromFacet(facet + 12);
            Vector3 v1 = parseVector3FromFacet(facet + 24);
            Vector3 v2 = parseVector3FromFacet(facet + 36);

            std::cerr << "v0: " << v0 << std::endl;
            std::cerr << "v1: " << v1 << std::endl;
            std::cerr << "v2: " << v2 << std::endl;

            std::cerr << std::endl;

            mesh->add_face(v0, v1, v2);



        }

    }

    return mesh;
}

#endif //RAYTRACER_FILEPARSER_H
