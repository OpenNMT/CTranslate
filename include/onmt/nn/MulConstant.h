#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename ModelT>
    class MulConstant: public Module<MatFwd>
    {
    public:
      MulConstant(th::Table* data);

      virtual MatFwd forward_impl(MatFwd& input) const;

    private:
      ModelT _scalar;
    };

  }
}

#include "onmt/nn/MulConstant.hxx"
