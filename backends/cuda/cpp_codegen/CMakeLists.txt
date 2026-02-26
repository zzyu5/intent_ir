cmake_minimum_required(VERSION 3.16)
project(intentir_cuda_codegen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(intentir_cuda_codegen
    intentir_cuda_codegen.cpp
    ir_model.cpp
    shape_eval.cpp
)
target_include_directories(intentir_cuda_codegen PRIVATE third_party)
target_compile_options(intentir_cuda_codegen PRIVATE -Wall -Wextra -Wno-unused-parameter)
