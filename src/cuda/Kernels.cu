#include "onmt/cuda/Kernels.cuh"

#include <cuda_runtime.h>

namespace onmt
{
  namespace cuda
  {
    namespace kernels
    {

      struct AddOp
      {
        __device__ __forceinline__ void operator()(float* out, const float* in)
        {
          *out += *in;
        }
      };

      template <typename Op>
      __global__ void pointwise2_kernel(float* __restrict__ dst,
                                        const float* __restrict__ src,
                                        int len)
      {
        int stride = gridDim.x * blockDim.x;
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        Op op;

        for (int i = tid; i < len; i += stride)
        {
          op(dst + i, src + i);
        }
      }

      template <typename Op>
      void pointwise2(float* dst, const float* src, int len)
      {
        int grid_size = -1;
        int block_size = -1;

        cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, &pointwise2_kernel<Op>);
        grid_size = (len + block_size - 1) / block_size;

        pointwise2_kernel<Op><<<grid_size, block_size>>>(dst, src, len);
      }

      void add(float* a, const float* b, int len)
      {
        pointwise2<AddOp>(a, b, len);
      }

    }
  }
}