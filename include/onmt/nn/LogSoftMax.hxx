#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    LogSoftMax<MatFwd>::LogSoftMax()
      : Module<MatFwd>("nn.LogSoftMax")
    {
    }

    template <typename MatFwd>
    MatFwd LogSoftMax<MatFwd>::forward_impl(MatFwd& input) const
    {
      MatFwd output(input.rows(), input.cols());

      for (int i = 0; i < input.rows(); ++i)
      {
        auto v = input.row(i);
        double max = v.maxCoeff();
        double log_z = log((v.array() - max).exp().sum()) + max;
        output.row(i).noalias() = (v.array() - log_z).matrix();
      }

      return output;
    }

  }
}
