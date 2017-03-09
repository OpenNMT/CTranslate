#pragma once

#include "onmt/nn/ModuleFactory.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Container: public Module<MatFwd>
    {
    public:
      Container(const std::string& name,
                th::Table* data,
                ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory);

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const = 0;

    protected:
      std::vector<Module<MatFwd>*> _sequence;
    };

  }
}

#include "onmt/nn/Container.hxx"
