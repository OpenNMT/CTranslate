#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class SelectTable: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      SelectTable(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.SelectTable")
        , _index(th::get_number(data, "index"))
      {
      }

      SelectTable(const SelectTable& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _index(other._index)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new SelectTable(*this);
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
