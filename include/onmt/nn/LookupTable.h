#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatEmb, typename ModelT>
    class LookupTable: public Module<MatFwd>
    {
    public:
      LookupTable(th::Table* data);

      virtual MatFwd forward_impl(MatFwd& input) const override;

    private:
      MatEmb _weight;
    };

  }
}

#include "onmt/nn/LookupTable.hxx"
