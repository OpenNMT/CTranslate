#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"
#include "onmt/simd/MatrixMult.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatEmb, typename ModelT>
    class qLookupTable: public Module<MatFwd>
    {
    public:
      qLookupTable(th::Table* data)
        : Module<MatFwd>("nn.qLookupTable")
        , _weightq(StorageLoader<Eigen::Map<const Eigen::RowMajorMat<short> >, short>::get_matrix(data, "weight"))
      {
        _weight.resize(_weightq.rows(), _weightq.cols());
        _weight.setZero();
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), _weight.cols());

        for (size_t i = 0; i < input.batches(); ++i) {
          if (_weight(input(i,0),0) == 0 && _weight(input(i,0),1) == 0 && _weight(input(i,0),2) == 0) {
            _weight.row(input(i,0)) = _weightq.row(input(i,0)).template cast<ModelT>() / simd::quant_mult;
          }
          this->_output.row(i).noalias() = _weight.row(input(i, 0));
        }
      }

    private:
      Eigen::Map<const Eigen::RowMajorMat<short> > _weightq;
      MatFwd _weight;
    };

  }
}
