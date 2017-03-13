#include "onmt/cuda/Kernels.cuh"

namespace onmt
{
  namespace cuda
  {

    template <typename T>
    __global__ void replicate_kernel(const T* vector, int len, T* matrix, int dim)
    {
      int idx = threadIdx.x + blockIdx.x * dim;
      T myval = vector[blockIdx.x];
      while (idx < ((blockIdx.x + 1) * dim))
      {
        matrix[idx] = myval;
        idx += blockDim.x;
      }
    }

    template <typename T>
    void replicate(const T* vector, int len, T* matrix, int dim)
    {
      replicate_kernel<<<len, 256>>>(vector, len, matrix, dim);
    }

    template void replicate<float>(const float* vector, int len, float* matrix, int dim);

  }
}