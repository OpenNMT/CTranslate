#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

#ifdef WITH_CUDA
#  include "onmt/cuda/Utils.h"
#endif

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class ModuleFactory
    {
    public:
      ModuleFactory(Profiler& profiler, bool cuda);
      ~ModuleFactory();

      Module<MatFwd>* build(th::Class* obj);

    private:
      std::vector<Module<MatFwd>*> _storage;
      Profiler& _profiler;
      bool _cuda;
#ifdef WITH_CUDA
      cublasHandle_t _handle;
#endif
    };

  }
}

#include "onmt/nn/ModuleFactory.hxx"
