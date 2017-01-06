#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Tanh<MatFwd>::Tanh()
      : Module<MatFwd>("nn.Tanh")
    {
    }

    template <typename MatFwd>
    MatFwd Tanh<MatFwd>::forward(MatFwd& input) const
    {
      return input.array().tanh().matrix();
    }

  }
}
