cmake_minimum_required(VERSION 3.21)
project(ai-model-cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYTORCH_ROOT "/opt/homebrew/opt/pytorch" CACHE PATH "Path to PyTorch")
set(TENSORFLOW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.11/site-packages/tensorflow" CACHE PATH "Path to tensorflow")

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


find_library(TENSORFLOW_CC_LIBRARY
    NAMES tensorflow_cc.2
    PATHS "${TENSORFLOW_ROOT}"
    REQUIRED)

find_library(TENSORFLOW_FRAMEWORK_LIBRARY
    NAMES tensorflow_framework.2
    PATHS "${TENSORFLOW_ROOT}"
    REQUIRED)

message(STATUS "TENSORFLOW_CC_LIBRARY: ${TENSORFLOW_CC_LIBRARY}")
message(STATUS "TENSORFLOW_FRAMEWORK_LIBRARY: ${TENSORFLOW_FRAMEWORK_LIBRARY}")

add_library(LibTensorflow INTERFACE)
target_link_libraries(LibTensorflow INTERFACE "${TENSORFLOW_FRAMEWORK_LIBRARY}" "${TENSORFLOW_CC_LIBRARY}" )
target_include_directories(LibTensorflow INTERFACE "${TENSORFLOW_ROOT}/include")

find_package(Eigen3 REQUIRED)

add_subdirectory(pytorch)
add_subdirectory(tensorflow)