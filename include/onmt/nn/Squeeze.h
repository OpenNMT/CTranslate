#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Squeeze: public Module<MatFwd>
    {
    public:
      Squeeze(th::Table* data)
        : Module<MatFwd>("nn.Squeeze")
        , _dimension(get_number(data, "dimension"))
      {
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = input;
        this->_output.squeeze(_dimension);
      }

    private:
      int _dimension;
    };

  }
}
