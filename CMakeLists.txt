cmake_minimum_required(VERSION 3.24)
project(tiny_llm_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 本机 GPU: RTX 4060 Laptop = sm_89。换机器时用
#   cmake -DCMAKE_CUDA_ARCHITECTURES=XX 覆盖
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/json.hpp)
  message(FATAL_ERROR
    "third_party/nlohmann/json.hpp 不存在，请先执行: python scripts/fetch_deps.py")
endif()

file(GLOB TLI_CORE_SOURCES CONFIGURE_DEPENDS
  src/config.cpp
  src/weights.cpp
  src/model.cpp
  src/sampler.cpp
  kernels/*.cu)

add_executable(tli src/main.cpp ${TLI_CORE_SOURCES})
target_include_directories(tli PRIVATE include third_party)
target_link_libraries(tli PRIVATE CUDA::cudart CUDA::cublas)
# 给 Nsight 用的行号信息
target_compile_options(tli PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(tli_test tests/test_forward.cpp ${TLI_CORE_SOURCES})
target_include_directories(tli_test PRIVATE include third_party)
target_link_libraries(tli_test PRIVATE CUDA::cudart CUDA::cublas)
