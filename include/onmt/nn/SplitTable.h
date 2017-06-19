#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class SplitTable: public Module<MatFwd>
    {
    public:
      SplitTable()
        : Module<MatFwd>("nn.SplitTable")
      {
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        // it is assumed that the previous reshape did the split
        this->_outputs = inputs;
      }
    };

  }
}
