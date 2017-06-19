#pragma once

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
        this->_output.resize(input.rows(), _weight.rows());
        this->_output = input * _weight.transpose();

        if (_bias.rows() > 0)
        {
          for (int i = 0; i < input.rows(); ++i)
            this->_output.row(i).noalias() += _bias.transpose();
        }
      }

      std::string get_details() const override
      {
        std::string details = std::to_string(_weight.cols()) + "->" + std::to_string(_weight.rows());
        if (_bias.rows() == 0)
          details += " without bias";
        return details;
      }

    protected:
      MatIn _weight;
      MatIn _bias;
    };

  }
}
