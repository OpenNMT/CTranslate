#pragma once

#include "onmt/nn/ModuleFactory.h"
#include "onmt/th/Utils.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Container: public Module<MatFwd>
    {
    public:
      Container(const std::string& name,
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

      /* apply recursively a generic function to each node of the graph */
      void* apply(void* (*func)(Module<MatFwd>*, void*), void* data)
      {
        for (auto child: _sequence)
          child->apply(func, data);
        return 0;
      }

      virtual void forward_impl(const std::vector<MatFwd>& inputs) = 0;

    protected:
      std::vector<Module<MatFwd>*> _sequence;
    };

  }
}
