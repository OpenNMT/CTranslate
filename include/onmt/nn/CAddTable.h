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

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) override
      {
        for (size_t i = input.size() - 1; i > 0; --i)
        {
          input[0] += input[i];
          input.pop_back();
        }

        return input;
      }

    };

  }
}
