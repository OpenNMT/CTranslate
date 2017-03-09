#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Sigmoid: public Module<MatFwd>
    {
    public:
      Sigmoid();

      virtual MatFwd forward_impl(MatFwd& input) const override;
    };

  }
}

#include "onmt/nn/Sigmoid.hxx"
