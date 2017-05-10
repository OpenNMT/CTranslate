#pragma once

#include "onmt/cuda/Utils.h"
#include "onmt/cuda/Kernels.cuh"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    cuLinear<MatFwd, MatIn, ModelT>::cuLinear(th::Table* data, cublasHandle_t& handle)
      : Linear<MatFwd, MatIn, ModelT>(data)
      , _handle(handle)
      , _bias_device(nullptr)
      , _weight_device(nullptr)
      , _input_device(nullptr)
      , _output_device(nullptr)
      , _expanded_bias_device(nullptr)
      , _allocated_batches(0)
    {
      // cuBLAS works with col-major matrices.
      _weight_device = cuda::to_device<float>(this->_weight.data(), this->_weight.cols(), this->_weight.rows());

      if (this->_bias.rows() > 0)
        _bias_device = cuda::to_device<float>(this->_bias.data(), this->_bias.rows());
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    cuLinear<MatFwd, MatIn, ModelT>::~cuLinear()
    {
      CUDA_CHECK(cudaFree(_weight_device));
      CUDA_CHECK(cudaFree(_bias_device));
      CUDA_CHECK(cudaFree(_input_device));
      CUDA_CHECK(cudaFree(_output_device));
      CUDA_CHECK(cudaFree(_expanded_bias_device));
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    void cuLinear<MatFwd, MatIn, ModelT>::realloc_device_buffers(int num_batches) const
    {
      CUDA_CHECK(cudaFree(_output_device));
      CUDA_CHECK(cudaFree(_input_device));
      CUDA_CHECK(cudaFree(_expanded_bias_device));

      _output_device = cuda::to_device<float>(this->_weight.rows(), num_batches);
      _input_device = cuda::to_device<float>(this->_weight.cols(), num_batches);

      if (_bias_device)
      {
        _expanded_bias_device = cuda::to_device<float>(this->_weight.rows(), num_batches);
        for (int i = 0; i < num_batches; ++i)
          CUDA_CHECK(cudaMemcpy(_expanded_bias_device + i * this->_weight.rows(),
                                _bias_device,
                                this->_weight.rows() * sizeof (float),
                                cudaMemcpyDeviceToDevice));

      }

      _allocated_batches = num_batches;
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    MatFwd cuLinear<MatFwd, MatIn, ModelT>::forward_impl(MatFwd& input) const
    {
      // See http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm

      const int batch_size = input.rows();
      const int input_size = input.cols();
      const int output_size = this->_weight.rows();

      if (batch_size > _allocated_batches)
        this->realloc_device_buffers(batch_size);

      cuda::to_device<float>(_input_device, input.data(), input_size, batch_size);

      float alpha = 1;
      float beta = 0;

      CUBLAS_CHECK(cublasSgemm(_handle,
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               output_size, batch_size, input_size,
                               &alpha,
                               _weight_device, input_size,
                               _input_device, input_size,
                               &beta,
                               _output_device, output_size));

      if (_expanded_bias_device)
        cuda::kernels::add(_output_device, _expanded_bias_device, batch_size * output_size);

      MatFwd output(batch_size, output_size);
      cuda::to_host<float>(_output_device, output.data(), output_size, batch_size);

      return output;
    }


  }
}
