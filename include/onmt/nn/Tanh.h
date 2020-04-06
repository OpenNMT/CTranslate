#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Tanh: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Tanh()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Tanh")
      {
      }

      Tanh(const Tanh& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Tanh(*this);
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = input.array().tanh().matrix();
      }
    };

  }
}
