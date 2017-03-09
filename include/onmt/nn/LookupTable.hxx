#pragma once

#include "onmt/StorageLoader.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatEmb, typename ModelT>
    LookupTable<MatFwd, MatEmb, ModelT>::LookupTable(th::Table* data)
      : Module<MatFwd>("nn.LookupTable")
      , _weight(StorageLoader<MatEmb, ModelT>::get_matrix(data, "weight"))
    {
    }

    template <typename MatFwd, typename MatEmb, typename ModelT>
    MatFwd LookupTable<MatFwd, MatEmb, ModelT>::forward_impl(MatFwd& input) const
    {
      MatFwd out(input.rows(), _weight.cols());

      for (size_t i = 0; i < input.batches(); ++i)
      {
        out.row(i).noalias() = _weight.row(input(i, 0));
      }

      return out;
    }

  }
}
