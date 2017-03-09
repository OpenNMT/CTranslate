#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    SplitTable<MatFwd>::SplitTable()
      : Module<MatFwd>("nn.SplitTable")
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> SplitTable<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      // it is assumed that the previous reshape did the split
      return input;
    }

  }
}
