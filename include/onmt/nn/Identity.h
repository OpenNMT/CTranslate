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

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        this->_outputs = inputs;
      }
    };

  }
}
