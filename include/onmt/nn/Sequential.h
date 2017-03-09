#pragma once

#include "onmt/nn/Container.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Sequential: public Container<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Sequential(th::Table* data, ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory);

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override;
    };

  }
}

#include "onmt/nn/Sequential.hxx"
