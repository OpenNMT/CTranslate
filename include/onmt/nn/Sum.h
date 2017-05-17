#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Sum: public Module<MatFwd>
    {
    public:
      Sum(th::Table* data)
        : Module<MatFwd>("nn.Sum")
        , _dimension(get_number(data, "dimension"))
      {
      }

      virtual MatFwd forward_impl(MatFwd& input) const override
      {
        return input.sum(_dimension);
      }

    private:
      int _dimension;
    };

  }
}
