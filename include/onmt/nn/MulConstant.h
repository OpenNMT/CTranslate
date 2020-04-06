#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class MulConstant: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      MulConstant(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.MulConstant")
        , _scalar(th::get_scalar<ModelT>(data, "constant_scalar"))
      {
      }

      MulConstant(const MulConstant& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _scalar(other._scalar)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new MulConstant(*this);
      }

      void forward_impl(const MatFwd& input)
      {
        this->_output.noalias() = input * _scalar;
      }

    private:
      ModelT _scalar;
    };

  }
}
