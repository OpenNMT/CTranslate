#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    SoftMax<MatFwd>::SoftMax()
      : Module<MatFwd>("nn.SoftMax")
    {
    }

    template <typename MatFwd>
    MatFwd SoftMax<MatFwd>::forward_impl(MatFwd& input) const
    {
      MatFwd output(input.rows(), input.cols());

      for (int i = 0; i < input.rows(); ++i)
      {
        auto v = input.row(i);
        double max = v.maxCoeff();
        output.row(i).noalias() =  ((v.array() - (log((v.array() - max).exp().sum()) + max)).exp()).matrix();
      }

      return output;
    }

  }
}
