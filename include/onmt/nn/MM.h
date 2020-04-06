#pragma once

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class MM: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      MM(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.MM")
        , _transA(get_boolean(data, "transA"))
        , _transB(get_boolean(data, "transB"))
      {
      }

      MM(const MM& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _transA(other._transA)
        , _transB(other._transB)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new MM(*this);
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        this->_output.resize(inputs[0].rows(), inputs[0].virtualRows()*inputs[1].virtualCols());
        this->_output.setHiddenDim(inputs[0].virtualRows());

        for (size_t i = 0; i < inputs[0].batches(); ++i)
        {
          MatFwd m1 = inputs[0].batch(i);
          MatFwd m2 = inputs[1].batch(i);

          if (_transA)
            m1.transposeInPlace();
          if (_transB)
            m2.transposeInPlace();

          MatFwd res = m1 * m2;

          this->_output.assign(i, res);
        }
      }

    private:
      bool _transA;
      bool _transB;
    };

  }
}
