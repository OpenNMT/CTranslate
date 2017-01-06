#pragma once

#include "onmt/nn/Container.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class ConcatTable: public Container<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      ConcatTable(th::Table* data);

      virtual std::vector<MatFwd> forward(std::vector<MatFwd>& input) const override;
    };

  }
}

#include "onmt/nn/ConcatTable.hxx"
