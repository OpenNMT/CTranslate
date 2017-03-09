#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Sigmoid<MatFwd>::Sigmoid()
      : Module<MatFwd>("nn.Sigmoid")
    {
    }

    template <typename MatFwd>
    MatFwd Sigmoid<MatFwd>::forward_impl(MatFwd& input) const
    {
      return (1.0 + (-input).array().exp()).inverse().matrix();
    }

  }
}
