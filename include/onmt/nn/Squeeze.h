#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Squeeze: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Squeeze(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Squeeze")
        , _dimension(get_number(data, "dimension"))
      {
      }

      Squeeze(const Squeeze& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _dimension(other._dimension)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Squeeze(*this);
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = input;
        this->_output.squeeze(_dimension);
      }

    private:
      int _dimension;
    };

  }
}
