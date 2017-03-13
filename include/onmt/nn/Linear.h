#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class Linear: public Module<MatFwd>
    {
    public:
      Linear(th::Table* data);

      virtual MatFwd forward_impl(MatFwd& input) const override;

      virtual std::string get_details() const override;

    private:
      MatIn _weight;
      MatIn _bias;
    };


#ifdef WITH_CUDA
    template <typename MatFwd, typename ModelT>
    class Linear<MatFwd, float*, ModelT>: public Module<MatFwd>
    {
    public:
      Linear(th::Table* data);
      virtual ~Linear();

      virtual MatFwd forward_impl(MatFwd& input) const override;

      virtual std::string get_details() const override;

    private:
      void realloc_output(int num_batches) const;

      int _input_size;
      int _output_size;

      float* _bias;
      float* _weight;

      // Reuse allocated gemm output.
      mutable float* _output;
      mutable size_t _allocated_batches;
    };
#endif

  }
}

#include "onmt/nn/Linear.hxx"
