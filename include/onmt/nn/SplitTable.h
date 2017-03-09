#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class SplitTable: public Module<MatFwd>
    {
    public:
      SplitTable();

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override;
    };

  }
}

#include "onmt/nn/SplitTable.hxx"
