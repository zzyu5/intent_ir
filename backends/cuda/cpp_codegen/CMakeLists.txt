cmake_minimum_required(VERSION 3.16)
project(intentir_cuda_codegen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(intentir_cuda_codegen intentir_cuda_codegen.cpp)
target_include_directories(intentir_cuda_codegen PRIVATE ../../spmd_rvv/cpp_codegen/third_party)
target_compile_options(intentir_cuda_codegen PRIVATE -Wall -Wextra -Wno-unused-parameter)

