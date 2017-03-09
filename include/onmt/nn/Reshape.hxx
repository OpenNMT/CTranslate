#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Reshape<MatFwd>::Reshape()
      : Module<MatFwd>("nn.Reshape")
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> Reshape<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      // also do the SplitTable
      std::vector<MatFwd> out;

      for (size_t i = 0; i < 4; ++i)
      {
        out.emplace_back(input[0].block(0, i*(input[0].cols()/4), input[0].rows(), input[0].cols()/4));
      }

      return out;
    }

  }
}
