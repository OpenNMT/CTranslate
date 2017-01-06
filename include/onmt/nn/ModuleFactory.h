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
      static void init();
      static void destroy();

      static Module<MatFwd>* build(th::Class* obj);

    private:
      static std::unordered_map<std::string, Module<MatFwd>*> _stateless_storage;
      static std::vector<Module<MatFwd>*> _storage;
    };

  }
}

#include "onmt/nn/ModuleFactory.hxx"
