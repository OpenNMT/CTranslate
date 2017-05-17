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

      std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) override
      {
        // it is assumed that the previous reshape did the split
        return input;
      }
    };

  }
}
