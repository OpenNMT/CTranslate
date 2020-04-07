#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class SplitTable: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      SplitTable()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.SplitTable")
      {
      }

      SplitTable(const SplitTable& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new SplitTable(*this);
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        // it is assumed that the previous reshape did the split
        this->_outputs = inputs;
      }
    };

  }
}
