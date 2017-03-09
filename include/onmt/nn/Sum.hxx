#pragma once

#include <iostream>

#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Sum<MatFwd>::Sum(th::Table* data)
      : Module<MatFwd>("nn.Sum")
      , _dimension(get_number(data, "dimension"))
    {
    }

    template <typename MatFwd>
    MatFwd Sum<MatFwd>::forward_impl(MatFwd& input) const
    {
      return input.sum(_dimension);
    }

  }
}
