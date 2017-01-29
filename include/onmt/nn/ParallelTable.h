#pragma once

#include "onmt/nn/Container.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class ParallelTable: public Container<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      ParallelTable(th::Table* data, ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory);

      virtual std::vector<MatFwd> forward(std::vector<MatFwd>& input) const override;
    };

  }
}

#include "onmt/nn/ParallelTable.hxx"
