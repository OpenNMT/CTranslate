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

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) override
      {
        std::vector<MatFwd> out;
        out.reserve(this->_sequence.size());

        for (auto& mod: this->_sequence)
        {
          std::vector<MatFwd> in(1, input[0]);
          out.push_back(mod->forward(in)[0]);
        }

        return out;
      }
    };

  }
}
