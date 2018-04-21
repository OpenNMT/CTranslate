#pragma once

#include "onmt/nn/Linear.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"
#include "onmt/simd/MatrixMult.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class qLinear: public Linear<MatFwd, MatIn, ModelT>
    {
    public:
      qLinear(th::Table* data)
        : Linear<MatFwd, MatIn, ModelT>(data)
      {
        // Quantize the weight - ncols=width is supposed to be multiple of SIMD_VSIZE
        if (this->_wcols % SIMD_VIZE)
          throw std::runtime_error("Weight matrix width should be multiple of 8/16 for qLinear");
        _quant_weight.resize(this->_wrows * this->_wcols / SIMD_VSIZE);
        Quantize(this->_weight.data(), _quant_weight.data(), this->_wrows, this->_wcols);
      }

      virtual ~qLinear()
      {
      }

      virtual void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), this->_wrows);

        _quant_input.resize(input.rows() * input.cols() / SIMD_VSIZE);

        Quantize(input.data(), _quant_input.data(), input.rows(), input.cols());

        SSE_MatrixMult(_quant_input.data(), _quant_weight.data(), this->_output.data(),
                       input.rows(), this->_wrows, this->_wcols);

        if (this->_bias.rows() > 0)
        {
          for (int i = 0; i < input.rows(); ++i)
            this->_output.row(i).noalias() += this->_bias.transpose();
        }
      }

      /* reduce a linear weigth matrix to a given vocabulary */
      virtual void apply_subdictionary(const std::vector<size_t>& v) override {
        _subdict = v;
      }

    protected:
      std::vector<SIMD_TYPE> _quant_weight;
      std::vector<SIMD_TYPE> _quant_input;
      std::vector<size_t> _subdict;
    };

  }
}
