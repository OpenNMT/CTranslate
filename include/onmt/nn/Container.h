#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Container: public Module<MatFwd>
    {
    public:
      Container(const std::string& name, th::Table* data);

      virtual std::vector<MatFwd> forward(std::vector<MatFwd>& input) const = 0;

    protected:
      std::vector<Module<MatFwd>*> _sequence;
    };

  }
}

#include "onmt/nn/Container.hxx"
