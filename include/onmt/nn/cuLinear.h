#pragma once

#include "onmt/nn/Linear.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class cuLinear: public Linear<MatFwd, MatIn, ModelT>
    {
    public:
      cuLinear(th::Table* data, cublasHandle_t& handle);
      ~cuLinear();

      virtual MatFwd forward_impl(MatFwd& input) const override;

    private:
      void realloc_device_buffers(int num_batches) const;

      cublasHandle_t& _handle;

      float* _bias_device;
      float* _weight_device;

      // Preallocate device buffers.
      mutable float* _input_device;
      mutable float* _output_device;
      mutable float* _expanded_bias_device;
      mutable size_t _allocated_batches;
    };

  }
}

#include "onmt/nn/cuLinear.hxx"
