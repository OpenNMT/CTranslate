#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class SoftMax: public Module<MatFwd>
    {
    public:
      SoftMax()
        : Module<MatFwd>("nn.SoftMax")
      {
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
