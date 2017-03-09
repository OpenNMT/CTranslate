#pragma once

#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    SelectTable<MatFwd>::SelectTable(th::Table* data)
      : Module<MatFwd>("nn.SelectTable")
      , _index(th::get_number(data, "index"))
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> SelectTable<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      int index;
      if (_index < 0)
        index = input.size() + _index;
      else
        index = _index - 1; // Lua is 1-indexed.

      std::vector<MatFwd> out;
      out.push_back(input[index]);
      return out;
    }

  }
}
