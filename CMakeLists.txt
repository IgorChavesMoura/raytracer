cmake_minimum_required(VERSION 3.17)
project(raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(., src)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
    add_executable(${PROJECT_NAME}_GPU src/gpu/main.cu)
    #if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #    target_compile_options(${PROJECT_NAME}_GPU $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    #endif()
    set_target_properties(
        ${PROJECT_NAME}_GPU
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "No CUDA compiler found")
endif()

add_executable(${PROJECT_NAME} src/main.cc)

file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
