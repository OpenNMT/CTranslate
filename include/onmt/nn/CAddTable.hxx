#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    CAddTable<MatFwd>::CAddTable()
      : Module<MatFwd>("nn.CAddTable")
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> CAddTable<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      input[0].noalias() += input[1];
      input.pop_back();
      return input;
    }

  }
}
