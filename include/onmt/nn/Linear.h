#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class Linear: public Module<MatFwd>
    {
    public:
      Linear(th::Table* data);
      virtual ~Linear() {};

      virtual MatFwd forward_impl(MatFwd& input) const override;

      virtual std::string get_details() const override;

    protected:
      MatIn _weight;
      MatIn _bias;
    };

  }
}

#include "onmt/nn/Linear.hxx"
