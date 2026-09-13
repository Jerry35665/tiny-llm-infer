#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t err__ = (expr);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d  %s -> %s\n", __FILE__,         \
                   __LINE__, #expr, cudaGetErrorString(err__));                 \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(expr)                                                      \
  do {                                                                          \
    cublasStatus_t st__ = (expr);                                               \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error at %s:%d  %s -> status %d\n",          \
                   __FILE__, __LINE__, #expr, (int)st__);                       \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)
