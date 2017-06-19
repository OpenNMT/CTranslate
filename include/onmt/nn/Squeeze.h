#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Squeeze: public Module<MatFwd>
    {
    public:
      Squeeze(th::Table* data);

      virtual MatFwd forward_impl(MatFwd& input) const override;

    private:
      int _dimension;
    };

  }
}

#include "onmt/nn/Squeeze.hxx"
