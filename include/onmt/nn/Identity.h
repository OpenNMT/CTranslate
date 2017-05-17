#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Identity: public Module<MatFwd>
    {
    public:
      Identity()
        : Module<MatFwd>("nn.Identity")
      {
      }

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) override
      {
        return input;
      }
    };

  }
}
