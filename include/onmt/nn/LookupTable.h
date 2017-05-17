#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatEmb, typename ModelT>
    class LookupTable: public Module<MatFwd>
    {
    public:
      LookupTable(th::Table* data)
        : Module<MatFwd>("nn.LookupTable")
        , _weight(StorageLoader<MatEmb, ModelT>::get_matrix(data, "weight"))
      {
      }

      virtual MatFwd forward_impl(MatFwd& input) const override
      {
        MatFwd out(input.rows(), _weight.cols());

        for (size_t i = 0; i < input.batches(); ++i)
        {
          out.row(i).noalias() = _weight.row(input(i, 0));
        }

        return out;
      }

    private:
      MatEmb _weight;
    };

  }
}
