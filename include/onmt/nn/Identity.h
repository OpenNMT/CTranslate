#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Identity: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Identity()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Identity")
      {
      }

      Identity(const Identity& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Identity(*this);
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        this->_outputs = inputs;
      }
    };

  }
}
