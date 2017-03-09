#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class LogSoftMax: public Module<MatFwd>
    {
    public:
      LogSoftMax();

      virtual MatFwd forward_impl(MatFwd& input) const;
    };

  }
}

#include "onmt/nn/LogSoftMax.hxx"
