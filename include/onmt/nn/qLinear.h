#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"
#include "onmt/simd/MatrixMult.h"

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
        // Quantize the weight - ncols=width is supposed to be multiple of 8
        MatIn _weight = StorageLoader<MatIn, ModelT>::get_matrix(data, "weight");
        _wrows = _weight.rows();
        _wcols = _weight.cols();
        if (_wcols % 8)
          throw std::runtime_error("Weight matrix width should be multiple of 8 for qLinear");
        _quant_weight.resize(_wrows * _wcols / 8);
        Quantize(_weight.data(), _quant_weight.data(), _wrows, _wcols);
      }

      virtual ~qLinear()
      {
      }

      virtual void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), _wrows);

        _quant_input.resize(input.rows() * input.cols() / 8);

        Quantize(input.data(), _quant_input.data(), input.rows(), input.cols());

        SSE_MatrixMult(_quant_input.data(), _quant_weight.data(), this->_output.data(),
                       input.rows(), _wrows, _wcols);

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

    protected:
      std::vector<__m128i> _quant_weight;
      std::vector<__m128i> _quant_input;
      size_t _wrows, _wcols;
      MatIn _bias;
    };

  }
}
