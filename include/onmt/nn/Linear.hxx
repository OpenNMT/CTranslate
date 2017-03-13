#pragma once

#include "onmt/StorageLoader.h"

#ifdef ANDROID_GNUSTL_COMPAT
#  include "onmt/android_gnustl_compat.h"
#endif

#ifdef WITH_CUDA
#  include "onmt/cuda/Utils.h"
#  include "onmt/cuda/Kernels.cuh"
#endif

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    Linear<MatFwd, MatIn, ModelT>::Linear(th::Table* data)
      : Module<MatFwd>("nn.Linear")
      , _weight(StorageLoader<MatIn, ModelT>::get_matrix(data, "weight"))
      , _bias(StorageLoader<MatIn, ModelT>::get_matrix(data, "bias"))
    {
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    MatFwd Linear<MatFwd, MatIn, ModelT>::forward_impl(MatFwd& input) const
    {
      input *= _weight.transpose();

      if (_bias.rows() > 0)
      {
        for (int i = 0; i < input.rows(); ++i)
          input.row(i) += _bias.transpose();
      }

      return input;
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    std::string Linear<MatFwd, MatIn, ModelT>::get_details() const
    {
      std::string details = std::to_string(_weight.cols()) + "->" + std::to_string(_weight.rows());
      if (_bias.rows() == 0)
        details += " without bias";
      return details;
    }


#ifdef WITH_CUDA
    template <typename MatFwd, typename ModelT>
    Linear<MatFwd, float*, ModelT>::Linear(th::Table* data)
      : Module<MatFwd>("nn.Linear")
      , _bias(StorageLoader<float*, ModelT>::get_matrix(data, "bias", _output_size, _input_size))
      , _weight(StorageLoader<float*, ModelT>::get_matrix(data, "weight", _output_size, _input_size))
      , _output(nullptr)
      , _allocated_batches(0)
    {
    }

    template <typename MatFwd, typename ModelT>
    Linear<MatFwd, float*, ModelT>::~Linear()
    {
      cudaFree(_weight);
      cudaFree(_bias);
      cudaFree(_output);
    }

    template <typename MatFwd, typename ModelT>
    void Linear<MatFwd, float*, ModelT>::realloc_output(int num_batches) const
    {
      cudaFree(_output);
      _output = cuda::to_device<float>(nullptr, num_batches, _output_size);
      _allocated_batches = num_batches;
    }

    template <typename MatFwd, typename ModelT>
    MatFwd Linear<MatFwd, float*, ModelT>::forward_impl(MatFwd& input) const
    {
      // See http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
      int m = input.rows();
      int n = _output_size;
      int k = input.cols();

      if (m > _allocated_batches)
      {
        this->realloc_output(m);
      }

      float* a = cuda::to_device<float>(input.data(), k, m);
      float* b = _weight;
      float* c = _output;

      float alpha = 1;
      float beta = 0;

      // Use c to add bias.
      if (_bias)
      {
        beta = 1;
        cuda::replicate(_bias, n, c, m);
      }

      cublasStatus_t status;
      status = cublasSgemm(*cuda::get_handle(),
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           m, n, k,
                           &alpha,
                           a, k,
                           b, k,
                           &beta,
                           c, m);

      if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemm failed");

      MatFwd output(n, m);
      cuda::to_host<float>(c, output.data(), m, n);
      output.transposeInPlace();

      cudaFree(a);

      return output;
    }

    template <typename MatFwd, typename ModelT>
    std::string Linear<MatFwd, float*, ModelT>::get_details() const
    {
      std::string details = std::to_string(_input_size) + "->" + std::to_string(_output_size);
      if (!_bias)
        details += " without bias";
      return details;
    }
#endif

  }
}
