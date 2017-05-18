#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class LogSoftMax: public Module<MatFwd>
    {
    public:
      LogSoftMax()
        : Module<MatFwd>("nn.LogSoftMax")
      {
      }

      void forward_impl(const MatFwd& input)
      {
        this->_output.resize(input.rows(), input.cols());

        for (int i = 0; i < input.rows(); ++i)
        {
          auto v = input.row(i);
          double max = v.maxCoeff();
          double log_z = log((v.array() - max).exp().sum()) + max;
          this->_output.row(i) = v.array() - log_z;
        }
      }
    };

  }
}
