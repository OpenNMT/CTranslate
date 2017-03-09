#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class MM: public Module<MatFwd>
    {
    public:
      MM(th::Table* data);

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override;

    private:
      bool _transA;
      bool _transB;
    };

  }
}

#include "onmt/nn/MM.hxx"
