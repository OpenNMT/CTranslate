#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class SoftMax: public Module<MatFwd>
    {
    public:
      SoftMax();

      virtual MatFwd forward(MatFwd& input) const;
    };

  }
}

#include "onmt/nn/SoftMax.hxx"
