#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class CAddTable: public Module<MatFwd>
    {
    public:
      CAddTable()
        : Module<MatFwd>("nn.CAddTable")
      {
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        this->_output = inputs[0];

        for (size_t i = 1; i < inputs.size(); ++i)
          this->_output.noalias() += inputs[i];
      }

    };

  }
}
