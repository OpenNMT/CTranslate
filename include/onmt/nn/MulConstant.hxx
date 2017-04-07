#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename ModelT>
    MulConstant<MatFwd, ModelT>::MulConstant(th::Table* data)
      : Module<MatFwd>("nn.MulConstant")
      , _scalar(th::get_scalar<ModelT>(data, "constant_scalar"))
    {
    }

    template <typename MatFwd, typename ModelT>
    MatFwd MulConstant<MatFwd, ModelT>::forward_impl(MatFwd& input) const
    {
      return input * _scalar;
    }

  }
}
