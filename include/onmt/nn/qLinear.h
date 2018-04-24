#pragma once

#include <memory>
#include "onmt/nn/Linear.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"
#include "onmt/Utils.h"
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
        : Linear<MatFwd, MatIn, ModelT>(data), _quant_input_buffer(nullptr)
      {
        // Quantize the weight - ncols=width is supposed to be multiple of SIMD_VSIZE
        if (this->_wcols % SIMD_VSIZE)
          throw std::runtime_error("Weight matrix width should be multiple of 8/16 for qLinear");
        _malloc_align(_quant_weight_buffer, _quant_weight, this->_wrows * this->_wcols / SIMD_VSIZE);
        Quantize(this->_weight.data(), _quant_weight, this->_wrows, this->_wcols);
      }

      virtual ~qLinear()
      {
        free(_quant_weight_buffer);
        free(_quant_input_buffer);
      }

      /* aligned allocation method - in c++17 we have aligned_alloc that we can use */
      void _malloc_align(void *&buffer, SIMD_TYPE *&data, size_t size)
      {
        buffer = nullptr;
        _realloc_align(buffer, data, size);
      }

      void _realloc_align(void *&buffer, SIMD_TYPE *&data, size_t size)
      {
        size_t buf_size = (size + 1) * sizeof(SIMD_TYPE);
        buffer = realloc(buffer, buf_size);
        if (!buffer)
          throw std::runtime_error("Cannot allocate memory");
        void* ptr = (void*)buffer;
        align(sizeof(SIMD_TYPE), size * sizeof(SIMD_TYPE), ptr, buf_size);
        data = reinterpret_cast<SIMD_TYPE*>(ptr);
      }

      virtual void forward_impl(const MatFwd& input) override
      {
        if (this->_rwrows)
          this->_output.resize(input.rows(), this->_rwrows);
        else
          this->_output.resize(input.rows(), this->_wrows);

        /* quantize the input */
        _realloc_align(_quant_input_buffer, _quant_input, input.rows() * input.cols() / SIMD_VSIZE);
        Quantize(input.data(), _quant_input, input.rows(), input.cols());

        SIMD_MatrixMult(_quant_input, _quant_weight, this->_output.data(),
                        input.rows(), (this->_rwrows?this->_rwrows:this->_wrows), this->_wcols,
                        _subdict);

        /* add bias */
        if (this->_bias.rows() > 0)
        {
          if (this->_rwrows)
            for (int i = 0; i < input.rows(); ++i)
              this->_output.row(i).noalias() += this->_rbias.transpose();
          else
            for (int i = 0; i < input.rows(); ++i)
              this->_output.row(i).noalias() += this->_bias.transpose();
        }
      }

      /* reduce a linear weigth matrix to a given vocabulary */
      virtual void apply_subdictionary(const std::vector<size_t>& v) override
      {
        this->_rwrows = v.size();
        _subdict = v;
        this->_rbias.resize(v.size(), 1);
        /* adjust bias */
        for (size_t i = 0; i < v.size(); i++) {
          this->_rbias.row(i) = this->_bias.row(v[i]);
        }
      }

    protected:
      void* _quant_weight_buffer;
      void* _quant_input_buffer;
      SIMD_TYPE* _quant_weight;
      SIMD_TYPE* _quant_input;
      std::vector<size_t> _subdict;
    };

  }
}
