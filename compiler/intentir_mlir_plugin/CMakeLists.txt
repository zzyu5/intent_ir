cmake_minimum_required(VERSION 3.20)

project(IntentIRMLIRPlugin LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The in-repo toolchain builds `mlir-opt` statically; pass plugins rely on
# symbol resolution against the host executable. Do not require all symbols to
# be resolved at link time for this shared library.
set(LLVM_NO_UNDEFINED_SYMBOLS OFF CACHE BOOL "" FORCE)

# Out-of-tree MLIR pass plugin.
#
# Note: Our in-repo toolchain (`artifacts/toolchains/mlir-current`) builds MLIR
# statically into `mlir-opt`. Pass plugins must share the *same* global pass
# registry instance as `mlir-opt`, so we intentionally allow unresolved MLIR/LLVM
# symbols in this shared library and rely on `mlir-opt` exporting them.

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(HandleLLVMOptions)

# LLVM's default CMake flags may enforce `-Wl,-z,defs` (no undefined symbols) for
# shared libraries. Pass plugins intentionally keep MLIR/LLVM symbols unresolved
# and rely on `mlir-opt`'s exported symbols at load time.
foreach(_flag_var IN ITEMS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
  if(DEFINED ${_flag_var})
    string(REPLACE "-Wl,-z,defs" "" ${_flag_var} "${${_flag_var}}")
  endif()
endforeach()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(IntentIRPasses SHARED
  IntentIRPasses.cpp
)

target_compile_features(IntentIRPasses PRIVATE cxx_std_17)
target_compile_options(IntentIRPasses PRIVATE -fno-exceptions -fno-rtti)
if(UNIX AND NOT APPLE)
  target_link_options(IntentIRPasses PRIVATE "-Wl,--allow-shlib-undefined")
endif()

set_target_properties(IntentIRPasses PROPERTIES
  OUTPUT_NAME "IntentIRPasses"
)
