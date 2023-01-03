cmake_minimum_required(VERSION 3.17)
project(RayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(., src)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
    add_executable(RayTracer_GPU src/gpu/main.cu)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(RayTracer_GPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
    set_target_properties(
            RayTracer_GPU
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "No CUDA compiler found")
endif()

add_executable(RayTracer src/main.cc)


