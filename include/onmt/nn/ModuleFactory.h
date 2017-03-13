#pragma once

#include <unordered_map>

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class ModuleFactory
    {
    public:
      ModuleFactory(Profiler& profiler);
      ~ModuleFactory();

      Module<MatFwd>* build(th::Class* obj);

    private:
      std::unordered_map<std::string, Module<MatFwd>*> _stateless_storage;
      std::vector<Module<MatFwd>*> _storage;
      Profiler& _profiler;
    };

  }
}

#include "onmt/nn/ModuleFactory.hxx"
