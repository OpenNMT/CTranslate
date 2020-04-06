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
      ModuleFactory(Profiler& profiler, bool cuda, bool qlinear);
      ModuleFactory(const ModuleFactory& other);
      ~ModuleFactory();

      size_t build(th::Class* obj);
      Module<MatFwd, MatIn, MatEmb, ModelT>* get_module(size_t id) const;
      void set_profiler(Profiler& profiler);

    private:
      std::vector<Module<MatFwd, MatIn, MatEmb, ModelT>*> _storage;
      Profiler* _profiler;
      bool _cuda;
      bool _qlinear;
#ifdef WITH_CUDA
      cublasHandle_t _handle;
#endif
    };

  }
}

#include "onmt/nn/ModuleFactory.hxx"
