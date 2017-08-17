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

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        if (this->_sequence.empty())
        {
          this->_outputs = inputs;
          return;
        }

        auto it = this->_sequence.begin();
        this->_outputs = (*it)->forward(inputs);

        for (it++; it != this->_sequence.end(); it++)
        {
          this->_outputs = (*it)->forward(this->_outputs);
        }
      }
    };

  }
}
