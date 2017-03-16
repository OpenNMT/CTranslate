#pragma once

#include "onmt/StorageLoader.h"

#ifdef ANDROID_GNUSTL_COMPAT
#  include "onmt/android_gnustl_compat.h"
#endif

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    Linear<MatFwd, MatIn, ModelT>::Linear(th::Table* data)
      : Module<MatFwd>("nn.Linear")
      , _weight(StorageLoader<MatIn, ModelT>::get_matrix(data, "weight"))
      , _bias(StorageLoader<MatIn, ModelT>::get_matrix(data, "bias"))
    {
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    MatFwd Linear<MatFwd, MatIn, ModelT>::forward_impl(MatFwd& input) const
    {
      input *= _weight.transpose();

      if (_bias.rows() > 0)
      {
        for (int i = 0; i < input.rows(); ++i)
          input.row(i) += _bias.transpose();
      }

      return input;
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    std::string Linear<MatFwd, MatIn, ModelT>::get_details() const
    {
      std::string details = std::to_string(_weight.cols()) + "->" + std::to_string(_weight.rows());
      if (_bias.rows() == 0)
        details += " without bias";
      return details;
    }

  }
}
