#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Replicate: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Replicate(th::Table* data)
        : Module<MatFwd, MatIn, MatEmb, ModelT>("nn.Replicate")
        , _dimension(th::get_number(data, "dim"))
        , _nfeatures(th::get_number(data, "nfeatures"))
      {
      }

      Replicate(const Replicate& other)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _dimension(other._dimension)
        , _nfeatures(other._nfeatures)
      {
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new Replicate(*this);
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output = input;

        if (_dimension == 2)
          this->_output.setHiddenDim(_nfeatures);
        else if (_dimension == 3)
          this->_output.setHiddenDim(input.cols());

        if (_nfeatures > 1)
          this->_output = input.replicate(1, _nfeatures);
      }

    private:
      int _dimension;
      int _nfeatures;
    };

  }
}
