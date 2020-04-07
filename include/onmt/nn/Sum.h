#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Sum: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Sum(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Sum")
        , _dimension(get_number(data, "dimension"))
      {
      }

      Sum(const Sum& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _dimension(other._dimension)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Sum(*this);
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = input.sum(_dimension);
      }

    private:
      int _dimension;
    };

  }
}
