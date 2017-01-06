#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Replicate: public Module<MatFwd>
    {
    public:
      Replicate(th::Table* data);

      virtual MatFwd forward(MatFwd& input) const override;

    private:
      int _dimension;
      int _nfeatures;
    };

  }
}

#include "onmt/nn/Replicate.hxx"
