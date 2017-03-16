#pragma once

#include "onmt/cuda/Utils.h"
#include "onmt/cuda/Kernels.cuh"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    LinearGPU<MatFwd, MatIn, ModelT>::LinearGPU(th::Table* data)
      : Linear<MatFwd, MatIn, ModelT>(data)
      , _bias_device(nullptr)
      , _weight_device(nullptr)
      , _output_device(nullptr)
      , _allocated_batches(0)
    {
      _weight_device = cuda::to_device<float>(this->_weight.data(), this->_weight.cols(), this->_weight.rows());

      if (this->_bias.rows() > 0)
        _bias_device = cuda::to_device<float>(this->_bias.data(), this->_bias.rows());
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    LinearGPU<MatFwd, MatIn, ModelT>::~LinearGPU()
    {
      CUDA_CHECK(cudaFree(_weight_device));
      CUDA_CHECK(cudaFree(_bias_device));
      CUDA_CHECK(cudaFree(_output_device));
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    void LinearGPU<MatFwd, MatIn, ModelT>::realloc_output(int num_batches) const
    {
      CUDA_CHECK(cudaFree(_output_device));
      _output_device = cuda::to_device<float>(nullptr, num_batches, this->_weight.rows());
      _allocated_batches = num_batches;
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    MatFwd LinearGPU<MatFwd, MatIn, ModelT>::forward_impl(MatFwd& input) const
    {
      // See http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
      int m = input.rows();
      int n = this->_weight.rows();
      int k = input.cols();

      if (m > _allocated_batches)
      {
        this->realloc_output(m);
      }

      float* a = cuda::to_device<float>(input.data(), k, m);
      float* b = _weight_device;
      float* c = _output_device;

      float alpha = 1;
      float beta = 0;

      // Use c to add bias.
      if (_bias_device)
      {
        beta = 1;
        cuda::replicate(_bias_device, n, c, m);
      }

      CUBLAS_CHECK(cublasSgemm(*cuda::get_handle(),
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               m, n, k,
                               &alpha,
                               a, k,
                               b, k,
                               &beta,
                               c, m));

      MatFwd output(n, m);
      cuda::to_host<float>(c, output.data(), m, n);
      output.transposeInPlace();

      CUDA_CHECK(cudaFree(a));

      return output;
    }


  }
}
