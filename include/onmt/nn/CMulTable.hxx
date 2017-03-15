#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    CMulTable<MatFwd>::CMulTable()
      : Module<MatFwd>("nn.CMulTable")
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> CMulTable<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      for (size_t i = input.size() - 1; i > 0; --i)
      {
        input[0] = input[0].cwiseProduct(input[i]);
        input.pop_back();
      }

      return input;
    }

  }
}
