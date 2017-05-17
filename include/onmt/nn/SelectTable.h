#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class SelectTable: public Module<MatFwd>
    {
    public:
      SelectTable(th::Table* data)
        : Module<MatFwd>("nn.SelectTable")
        , _index(th::get_number(data, "index"))
      {
      }

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override
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

    private:
      int _index;
    };

  }
}
