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

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        // Compute final size.
        int rows = inputs[0].rows();
        int cols = 0;

        for (size_t i = 0; i < inputs.size(); ++i)
          cols += inputs[i].cols();

        this->_output.resize(rows, cols);

        // Join column-wise by default.
        int offset = 0;

        for (size_t i = 0; i < inputs.size(); ++i)
        {
          this->_output.block(0, offset, rows, inputs[i].cols()) = inputs[i];
          offset += inputs[i].cols();
        }
      }
    };

  }
}
