#pragma once

#include <memory>
#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"

#ifdef ANDROID_GNUSTL_COMPAT
#  include "onmt/android_gnustl_compat.h"
#endif

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Linear: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Linear(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Linear")
        , _weight(new MatIn(StorageLoader<MatIn, ModelT>::get_matrix(data, "weight")))
        , _bias(new MatIn(StorageLoader<MatIn, ModelT>::get_matrix(data, "bias")))
      {
        _wrows = _weight->rows();
        _wcols = _weight->cols();
        _rwrows = 0;
      }

      Linear(const Linear& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _weight(other._weight)
        , _bias(other._bias)
      {
        _wrows = _weight->rows();
        _wcols = _weight->cols();
        _rwrows = 0;
      }

      virtual ~Linear()
      {
      }

      virtual Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Linear(*this);
      }

      virtual void forward_impl(const MatFwd& input) override
      {
        if (_rwrows)
        {
          this->_output.resize(input.rows(), _rwrows);
          this->_output = input * _rweight.transpose();
          if (_rbias.rows() > 0)
          {
            for (int i = 0; i < input.rows(); ++i)
              this->_output.row(i).noalias() += _rbias.transpose();
          }
        }
        else
        {
          this->_output.resize(input.rows(), _wrows);
          this->_output = input * _weight->transpose();
          if (_bias->rows() > 0)
          {
            for (int i = 0; i < input.rows(); ++i)
              this->_output.row(i).noalias() += _bias->transpose();
          }
        }
      }

      std::string get_details() const override
      {
        std::string details = std::to_string(_wcols) + "->" + std::to_string(_wrows);
        if (_bias->rows() == 0)
          details += " without bias";
        return details;
      }

      size_t get_weight_rows() const
      {
        return _wrows;
      }

      /* reduce the weight matrix to a given vocabulary, v is the list of index to keep */
      virtual void apply_subdictionary(const std::vector<size_t>& v)
      {
        _rwrows = v.size();
        _rweight.resize(v.size(), _wcols);
        _rbias.resize(v.size(), 1);
        /* build sub-matrix */
        for (size_t i = 0; i < v.size(); i++)
        {
          _rweight.row(i) = _weight->row(v[i]);
          _rbias.row(i) = _bias->row(v[i]);
        }
      }

    protected:
      std::shared_ptr<MatIn> _weight;
      std::shared_ptr<MatIn> _bias;
      Eigen::RowMajorMat<ModelT> _rweight;
      Eigen::RowMajorMat<ModelT> _rbias;
      size_t _wrows, _wcols;
      size_t _rwrows;
    };

  }
}
