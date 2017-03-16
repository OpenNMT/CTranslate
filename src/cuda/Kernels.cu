#include "onmt/cuda/Kernels.cuh"

namespace onmt
{
  namespace cuda
  {

    // This kernel replicates the vector `vector` in the matrix `matrix`.
    // Credits to Robert Crovella (http://stackoverflow.com/a/25453485).
    template <typename T>
    __global__ void replicate_kernel(const T* vector, int len, T* matrix, int rows, int cols)
    {
      int dim;

      if (len == cols)
      {
        dim = rows;
        int idx = threadIdx.x + blockIdx.x * dim;
        T myval = vector[blockIdx.x];
        while (idx < ((blockIdx.x + 1) * dim))
        {
          matrix[idx] = myval;
          idx += blockDim.x;
        }
      }
      else
      {
        dim = cols;
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        T myval = vector[idx % len];
        while (idx < dim * len){
          matrix[idx] = myval;
          idx += gridDim.x * blockDim.x;
        }
      }
    }

    template <typename T>
    void replicate(const T* vector, int len, T* matrix, int rows, int cols)
    {
      int blocks = 500;
      if (len == cols)
        blocks = len;

      replicate_kernel<<<blocks, 512>>>(vector, len, matrix, rows, cols);
    }

    template void replicate<float>(const float* vector, int len, float* matrix, int rows, int cols);

  }
}
