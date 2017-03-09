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
      Sum(th::Table* data);

      virtual MatFwd forward_impl(MatFwd& input) const override;

    private:
      int _dimension;
    };

  }
}

#include "onmt/nn/Sum.hxx"
