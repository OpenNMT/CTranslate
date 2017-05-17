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

      virtual MatFwd forward_impl(MatFwd& input)
      {
        MatFwd output(input.rows(), input.cols());

        for (int i = 0; i < input.rows(); ++i)
        {
          auto v = input.row(i);
          double max = v.maxCoeff();
          output.row(i).noalias() =  ((v.array() - (log((v.array() - max).exp().sum()) + max)).exp()).matrix();
        }

        return output;
      }
    };

  }
}
