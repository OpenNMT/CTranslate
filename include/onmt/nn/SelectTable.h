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

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        int index;
        if (_index < 0)
          index = inputs.size() + _index;
        else
          index = _index - 1; // Lua is 1-indexed.

        this->_output = inputs[index];
      }

    private:
      int _index;
    };

  }
}
