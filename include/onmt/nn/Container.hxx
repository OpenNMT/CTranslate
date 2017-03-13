#pragma once

#include "onmt/nn/ModuleFactory.h"
#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Container<MatFwd, MatIn, MatEmb, ModelT>::Container(const std::string& name,
                                                        th::Table* data,
                                                        ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
      : Module<MatFwd>(name, false)
    {
      th::Table* modules = th::get_field<th::Table*>(data, "modules");
      _sequence.reserve(modules->get_array().size());

      for (auto module_obj: modules->get_array())
      {
        th::Class* module = dynamic_cast<th::Class*>(module_obj);
        _sequence.push_back(factory.build(module));
      }
    }

  }
}
