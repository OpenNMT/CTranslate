#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"
#include "onmt/SSE_Matrix_Mult.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class qLinear: public Module<MatFwd>
    {
    public:
      qLinear(th::Table* data)
        : Module<MatFwd>("nn.qLinear")
        , _bias(StorageLoader<MatIn, ModelT>::get_matrix(data, "bias"))
      {
        // Quantize the weight - ncols=width is suppose to be multiple of 8
        MatIn _weight = StorageLoader<MatIn, ModelT>::get_matrix(data, "weight");
        _wrows = _weight.rows();
        _wcols = _weight.cols();
        if (_wcols % 8)
          throw std::runtime_error("Weight matrix width should be multiple of 8 for qLinear");
        _quant_weight.resize(_wrows * _wcols / 8);
        Quantize(_weight.data(), _quant_weight.data(), _quant_mult, _wrows, _wcols);
      }

      virtual ~qLinear()
      {
      }

      virtual void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), _wrows);

        _quant_input.resize(input.rows() * input.cols() / 8);

        Quantize(input.data(), _quant_input.data(), _quant_mult, input.rows(), input.cols());

        SSE_MatrixMult(_quant_input.data(), _quant_weight.data(), this->_output.data(),
                       _unquant_mult, input.rows(), _wrows, _wcols);

        if (_bias.rows() > 0)
        {
          for (int i = 0; i < input.rows(); ++i)
            this->_output.row(i).noalias() += _bias.transpose();
        }
      }

      std::string get_details() const override
      {
        std::string details = std::to_string(_wcols) + "->" + std::to_string(_wrows);
        if (_bias.rows() == 0)
          details += " without bias";
        return details;
      }

    // We quantize with 10 bits of precision. This works well "universally".
    // See the top of this file for more info on why.
    //double quant_mult = pow(2.0, 10.0);
    const float _quant_mult = 1000.0;

    // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
    // So we must divide by 1.0/(n^2) to get back the original value.
    const float _unquant_mult = 1.0/(_quant_mult*_quant_mult);

    protected:
      std::vector<__m128i> _quant_weight;
      std::vector<__m128i> _quant_input;
      size_t _wrows, _wcols;
      MatIn _bias;
    };

  }
}
