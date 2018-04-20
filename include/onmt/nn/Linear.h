#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"
#include <vector>

#ifdef ANDROID_GNUSTL_COMPAT
#  include "onmt/android_gnustl_compat.h"
#endif

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class Linear: public Module<MatFwd>
    {
    public:
      Linear(th::Table* data)
        : Module<MatFwd>("nn.Linear")
        , _weight(StorageLoader<MatIn, ModelT>::get_matrix(data, "weight"))
        , _bias(StorageLoader<MatIn, ModelT>::get_matrix(data, "bias"))
      {
      }

      virtual ~Linear()
      {
      }

      virtual void forward_impl(const MatFwd& input) override
      {
        if (_rweight.rows()) {
          this->_output.resize(input.rows(), _rweight.rows());
          this->_output = input * _rweight.transpose();
          if (_bias.rows() > 0)
          {
            for (int i = 0; i < input.rows(); ++i)
              this->_output.row(i).noalias() += _rbias.transpose();
          }
        } else {
          this->_output.resize(input.rows(), _weight.rows());
          this->_output = input * _weight.transpose();
          if (_bias.rows() > 0)
          {
            for (int i = 0; i < input.rows(); ++i)
              this->_output.row(i).noalias() += _bias.transpose();
          }
        }

      }

      std::string get_details() const override
      {
        std::string details = std::to_string(_weight.cols()) + "->" + std::to_string(_weight.rows());
        if (_bias.rows() == 0)
          details += " without bias";
        return details;
      }

      const MatIn &getWeight() { return _weight; }
      const MatIn &getBias() { return _bias; }
      Eigen::RowMajorMat<ModelT> &getRWeight() { return _rweight; }
      Eigen::RowMajorMat<ModelT> &getRBias() { return _rbias; }

    protected:
      MatIn _weight;
      MatIn _bias;
      Eigen::RowMajorMat<ModelT> _rweight;
      Eigen::RowMajorMat<ModelT> _rbias;
    };

  }
}
