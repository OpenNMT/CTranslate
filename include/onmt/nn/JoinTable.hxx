#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    JoinTable<MatFwd>::JoinTable()
      : Module<MatFwd>("nn.JoinTable")
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> JoinTable<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> out;
      out.emplace_back(input[0].rows(), input[0].cols() + input[1].cols());
      out.back() << input[0], input[1];
      return out;
    }

  }
}
