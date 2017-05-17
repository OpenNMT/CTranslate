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

      virtual MatFwd forward_impl(MatFwd& input) const override
      {
        return (1.0 + (-input).array().exp()).inverse().matrix();
      }
    };

  }
}
