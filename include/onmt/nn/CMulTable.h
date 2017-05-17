#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class CMulTable: public Module<MatFwd>
    {
    public:
      CMulTable()
        : Module<MatFwd>("nn.CMulTable")
      {
      }

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override
      {
        for (size_t i = input.size() - 1; i > 0; --i)
        {
          input[0] = input[0].cwiseProduct(input[i]);
          input.pop_back();
        }

        return input;
      }
    };

  }
}
