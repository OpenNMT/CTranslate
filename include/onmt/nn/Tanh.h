#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Tanh: public Module<MatFwd>
    {
    public:
      Tanh()
        : Module<MatFwd>("nn.Tanh")
      {
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = input.array().tanh().matrix();
      }
    };

  }
}
