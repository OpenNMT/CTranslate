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
      Reshape(th::Table* data)
        : Module<MatFwd>("nn.Reshape")
        , _dims(th::get_storage_as_vector<long>(data, "size"))
      {
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        // also do the SplitTable
        long leading_dim = _dims[0];

        this->_outputs.resize(leading_dim);

        for (long i = 0; i < leading_dim; ++i)
        {
          this->_outputs[i] = inputs[0].block(0,
                                             i * (inputs[0].cols() / leading_dim),
                                             inputs[0].rows(),
                                             inputs[0].cols() / leading_dim);
        }
      }

    private:
      std::vector<long> _dims;
    };

  }
}
