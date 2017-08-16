#pragma once

#include "onmt/nn/Container.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class ConcatTable: public Container<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      ConcatTable(th::Table* data, ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
        : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.ConcatTable", data, factory)
      {
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        this->_outputs.resize(this->_sequence.size());

        for (size_t i = 0; i < this->_sequence.size(); ++i)
          this->_outputs[i] = this->_sequence[i]->forward(inputs)[0];
      }
    };

  }
}
