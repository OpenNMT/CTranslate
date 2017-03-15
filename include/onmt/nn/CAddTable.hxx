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
      for (size_t i = input.size() - 1; i > 0; --i)
      {
        input[0] += input[i];
        input.pop_back();
      }

      return input;
    }

  }
}
