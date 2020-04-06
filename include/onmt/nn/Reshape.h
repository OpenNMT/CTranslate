#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Reshape: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Reshape(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Reshape")
        , _dims(th::get_storage_as_vector<long>(data, "size"))
      {
      }

      Reshape(const Reshape& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _dims(other._dims)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Reshape(*this);
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
