#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class SelectTable: public Module<MatFwd>
    {
    public:
      SelectTable(th::Table* data);

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override;

    private:
      int _index;
    };

  }
}

#include "onmt/nn/SelectTable.hxx"
