#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class JoinTable: public Module<MatFwd>
    {
    public:
      JoinTable()
        : Module<MatFwd>("nn.JoinTable")
      {
      }

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) override
      {
        std::vector<MatFwd> out;

        // Compute final size.
        int rows = input[0].rows();
        int cols = 0;

        for (size_t i = 0; i < input.size(); ++i)
          cols += input[i].cols();

        out.emplace_back(rows, cols);

        // Join column-wise by default.
        int offset = 0;

        for (size_t i = 0; i < input.size(); ++i)
        {
          out.back().block(0, offset, rows, input[i].cols()) = input[i];
          offset += input[i].cols();
        }

        return out;
      }
    };

  }
}
