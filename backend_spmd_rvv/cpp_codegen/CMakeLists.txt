cmake_minimum_required(VERSION 3.16)
project(intentir_codegen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(intentir_codegen intentir_codegen.cpp)
target_include_directories(intentir_codegen PRIVATE third_party)
target_compile_options(intentir_codegen PRIVATE -Wall -Wextra -Wno-unused-parameter)

