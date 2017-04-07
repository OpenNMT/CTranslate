#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Reshape: public Module<MatFwd>
    {
    public:
      Reshape(th::Table* data);

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override;

    private:
      std::vector<long> _dims;
    };

  }
}

#include "onmt/nn/Reshape.hxx"
