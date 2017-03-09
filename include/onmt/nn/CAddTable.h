#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class CAddTable: public Module<MatFwd>
    {
    public:
      CAddTable();

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override;
    };

  }
}

#include "onmt/nn/CAddTable.hxx"
