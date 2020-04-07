#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class SoftMax: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      SoftMax()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.SoftMax")
      {
      }

      SoftMax(const SoftMax& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new SoftMax(*this);
      }

      void forward_impl(const MatFwd& input)
      {
        this->_output.resizeLike(input);

        for (int i = 0; i < input.rows(); ++i)
        {
          auto v = input.row(i);
          double max = v.maxCoeff();
          this->_output.row(i) = ((v.array() - (log((v.array() - max).exp().sum()) + max)).exp());
        }
      }
    };

  }
}
