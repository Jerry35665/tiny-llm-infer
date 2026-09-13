// RMSNorm —— 参考实现。后续所有算子照这个风格写：朴素、正确、可对拍。
// 数学：out[r, :] = x[r, :] / sqrt(mean(x[r,:]^2) + eps) * w[:]
// 布局：一个 block 负责一行（rows == gridDim.x），block 内共享内存两段规约。
#include "tli/cuda_check.h"
#include "tli/kernels.h"

namespace tli {

namespace {

// block 内求和规约：warp shuffle + shared memory。
// M1 阶段刻意用最直白的写法（最后一个 warp 由 thread0 串行累加），
// 优先保证正确性；M3 优化时你会回来重写它。
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];
  __shared__ float block_sum;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();

  if (threadIdx.x == 0) {
    float s = 0.f;
    const int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) s += warp_sums[i];
    block_sum = s;
  }
  __syncthreads();
  return block_sum;
}

__global__ void rmsnorm_kernel(float* __restrict__ out,
                               const float* __restrict__ x,
                               const float* __restrict__ w, int hidden,
                               float eps) {
  const float* xr = x + (size_t)blockIdx.x * hidden;
  float* orow = out + (size_t)blockIdx.x * hidden;

  float local = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) local += xr[i] * xr[i];
  const float sum2 = block_reduce_sum(local);

  const float inv = rsqrtf(sum2 / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    orow[i] = xr[i] * inv * w[i];
}

}  // namespace

void rmsnorm(float* out, const float* x, const float* w, int rows, int hidden,
             float eps, cudaStream_t stream) {
  rmsnorm_kernel<<<rows, 256, 0, stream>>>(out, x, w, hidden, eps);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace tli
