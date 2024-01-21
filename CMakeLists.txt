cmake_minimum_required(VERSION 3.21)
project(ai-model-cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYTORCH_ROOT "/opt/homebrew/opt/pytorch" CACHE PATH "Path to PyTorch")


find_library(TORCH_CPU_LIBRARY
    NAMES torch_cpu
    PATHS "${PYTORCH_ROOT}/lib"
    REQUIRED)
find_library(TORCH_LIBRARY
    NAMES torch
    PATHS "${PYTORCH_ROOT}/lib"
    REQUIRED)
find_library(TORCH_C10_LIBRARY
    NAMES c10
    PATHS "${PYTORCH_ROOT}/lib"
    REQUIRED)

add_library(PyTorch INTERFACE)
target_link_libraries(PyTorch INTERFACE "${TORCH_CPU_LIBRARY}" "${TORCH_LIBRARY}" "${TORCH_C10_LIBRARY}")
target_include_directories(PyTorch INTERFACE "${PYTORCH_ROOT}/include")

get_target_property(AAA PyTorch INCLUDE_DIRECTORIES)

find_package(Eigen3 REQUIRED)

add_subdirectory(pytorch)
