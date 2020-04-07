#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class JoinTable: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      JoinTable()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.JoinTable")
      {
      }

      JoinTable(const JoinTable& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new JoinTable(*this);
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
