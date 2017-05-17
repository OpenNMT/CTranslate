#pragma once

#include "onmt/nn/Container.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Sequential: public Container<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Sequential(th::Table* data, ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
        : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.Sequential", data, factory)
      {
      }

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override
      {
        if (this->_sequence.empty())
          return input;

        auto it = this->_sequence.begin();
        std::vector<MatFwd> out = (*it)->forward(input);

        for (it++; it != this->_sequence.end(); it++)
        {
          out = (*it)->forward(out);
        }

        return out;
      }
    };

  }
}
