#pragma once

#include <memory>
#include "onmt/nn/ModuleFactory.h"
#include "onmt/th/Utils.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Container: public Module<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      Container(const std::string& name,
                th::Table* data,
                ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(name, false)
        , _factory(factory)
        , _sequence(new std::vector<size_t>())
      {
        th::Table* modules = th::get_field<th::Table*>(data, "modules");
        _sequence->reserve(modules->get_array().size());

        for (auto module_obj: modules->get_array())
        {
          th::Class* module = dynamic_cast<th::Class*>(module_obj);
          _sequence->push_back(factory.build(module));
        }
      }

      Container(const Container& other,
                const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
        : Module<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _factory(factory)
        , _sequence(other._sequence)
      {
      }

      /* apply recursively a generic function to each node of the graph */
      void* apply(void* (*func)(Module<MatFwd, MatIn, MatEmb, ModelT>*, void*), void* data)
      {
        func(this, data);

        for (auto child: *_sequence)
          _factory.get_module(child)->apply(func, data);
        return 0;
      }

      virtual void forward_impl(const std::vector<MatFwd>& inputs) = 0;

    protected:
      const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& _factory;
      std::shared_ptr<std::vector<size_t>> _sequence;
    };

  }
}
