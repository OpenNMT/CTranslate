#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class LogSoftMax: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      LogSoftMax()
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.LogSoftMax")
      {
      }

      LogSoftMax(const LogSoftMax& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new LogSoftMax(*this);
      }

      void forward_impl(const MatFwd& input)
      {
        this->_output.resize(input.rows(), input.cols());

        for (int i = 0; i < input.rows(); ++i)
        {
          auto v = input.row(i);
          double max = v.maxCoeff();
          double log_z = log((v.array() - max).exp().sum()) + max;
          this->_output.row(i) = v.array() - log_z;
        }
      }
    };

  }
}
