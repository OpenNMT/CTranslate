#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Tanh: public Module<MatFwd>
    {
    public:
      Tanh();

      virtual MatFwd forward_impl(MatFwd& input) const override;
    };

  }
}

#include "onmt/nn/Tanh.hxx"
