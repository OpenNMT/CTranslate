#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Sigmoid: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Sigmoid()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Sigmoid")
      {
      }

      Sigmoid(const Sigmoid& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Sigmoid(*this);
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = (1.0 + (-input).array().exp()).inverse();
      }
    };

  }
}
