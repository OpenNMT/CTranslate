#pragma once

#include <memory>
#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class LookupTable: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      LookupTable(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.LookupTable")
        , _weight(new MatEmb(StorageLoader<MatEmb, ModelT>::get_matrix(data, "weight")))
      {
      }

      LookupTable(const LookupTable& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _weight(other._weight)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new LookupTable(*this);
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), _weight->cols());

        for (size_t i = 0; i < input.batches(); ++i)
          this->_output.row(i).noalias() = _weight->row(input(i, 0));
      }

    private:
      std::shared_ptr<MatEmb> _weight;
    };

  }
}
