#pragma once

#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Squeeze<MatFwd>::Squeeze(th::Table* data)
      : Module<MatFwd>("nn.Squeeze")
      , _dimension(get_number(data, "dimension"))
    {
    }

    template <typename MatFwd>
    MatFwd Squeeze<MatFwd>::forward_impl(MatFwd& input) const
    {
      input.squeeze(_dimension);
      return input;
    }

  }
}
