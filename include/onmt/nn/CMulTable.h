#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class CMulTable: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      CMulTable()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.CMulTable")
      {
      }

      CMulTable(const CMulTable& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new CMulTable(*this);
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        this->_output = inputs[0];

        for (size_t i = 1; i < inputs.size(); ++i)
          this->_output.noalias() = this->_output.cwiseProduct(inputs[i]);
      }
    };

  }
}
