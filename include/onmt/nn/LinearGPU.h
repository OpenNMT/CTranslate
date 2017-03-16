#pragma once

#include "onmt/nn/Linear.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class LinearGPU: public Linear<MatFwd, MatIn, ModelT>
    {
    public:
      LinearGPU(th::Table* data);
      ~LinearGPU();

      virtual MatFwd forward_impl(MatFwd& input) const override;

    private:
      void realloc_output(int num_batches) const;

      float* _bias_device;
      float* _weight_device;

      // Reuse allocated gemm output.
      mutable float* _output_device;
      mutable size_t _allocated_batches;
    };

  }
}

#include "onmt/nn/LinearGPU.hxx"
