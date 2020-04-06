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

      Sequential(const Sequential& other,
                 const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
        : Container<MatFwd, MatIn, MatEmb, ModelT>(other, factory)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>* factory) const override
      {
        return new Sequential(*this, *factory);
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        if (this->_sequence->empty())
        {
          this->_outputs = inputs;
          return;
        }

        auto it = this->_sequence->begin();
        this->_outputs = this->_factory.get_module(*it)->forward(inputs);

        for (it++; it != this->_sequence->end(); it++)
        {
          this->_outputs = this->_factory.get_module(*it)->forward(this->_outputs);
        }
      }
    };

  }
}
