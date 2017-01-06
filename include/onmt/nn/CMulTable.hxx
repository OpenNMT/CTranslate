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
    std::vector<MatFwd> CMulTable<MatFwd>::forward(std::vector<MatFwd>& input) const
    {
      input[0] = input[0].cwiseProduct(input[1]);
      input.pop_back();
      return Module<MatFwd>::wrap_return(input);
    }

  }
}
