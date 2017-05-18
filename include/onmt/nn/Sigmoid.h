#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Sigmoid: public Module<MatFwd>
    {
    public:
      Sigmoid()
        : Module<MatFwd>("nn.Sigmoid")
      {
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = (1.0 + (-input).array().exp()).inverse();
      }
    };

  }
}
