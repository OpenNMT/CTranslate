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

      void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), _weight.cols());

        for (size_t i = 0; i < input.batches(); ++i)
          this->_output.row(i).noalias() = _weight.row(input(i, 0));
      }

    private:
      MatEmb _weight;
    };

  }
}
