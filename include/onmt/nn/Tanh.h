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

      virtual MatFwd forward_impl(MatFwd& input) override
      {
        return input.array().tanh().matrix();
      }
    };

  }
}
