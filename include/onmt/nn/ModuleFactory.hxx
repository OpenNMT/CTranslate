#pragma once

#include <iostream>

#include "onmt/nn/Linear.h"
#include "onmt/nn/LookupTable.h"
#include "onmt/nn/CAddTable.h"
#include "onmt/nn/CMulTable.h"
#include "onmt/nn/Sigmoid.h"
#include "onmt/nn/Tanh.h"
#include "onmt/nn/SplitTable.h"
#include "onmt/nn/JoinTable.h"
#include "onmt/nn/SelectTable.h"
#include "onmt/nn/Reshape.h"
#include "onmt/nn/Replicate.h"
#include "onmt/nn/Identity.h"
#include "onmt/nn/SoftMax.h"
#include "onmt/nn/LogSoftMax.h"
#include "onmt/nn/MM.h"
#include "onmt/nn/Sum.h"

#include "onmt/nn/Sequential.h"
#include "onmt/nn/ParallelTable.h"
#include "onmt/nn/ConcatTable.h"

#include "onmt/nn/Graph.h"

namespace onmt
{
  namespace nn
  {


    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::ModuleFactory()
    {
      // These modules are stateless so we can reuse the same instance for different
      // nodes in the graph.
      _stateless_storage["nn.CAddTable"] = new CAddTable<MatFwd>();
      _stateless_storage["nn.CMulTable"] = new CMulTable<MatFwd>();
      _stateless_storage["nn.Sigmoid"] = new Sigmoid<MatFwd>();
      _stateless_storage["nn.Tanh"] = new Tanh<MatFwd>();
      _stateless_storage["nn.SplitTable"] = new SplitTable<MatFwd>();
      _stateless_storage["nn.JoinTable"] = new JoinTable<MatFwd>();
      _stateless_storage["nn.Reshape"] = new Reshape<MatFwd>();
      _stateless_storage["nn.SoftMax"] = new SoftMax<MatFwd>();
      _stateless_storage["nn.LogSoftMax"] = new LogSoftMax<MatFwd>();
      _stateless_storage["nn.Identity"] = new Identity<MatFwd>();
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::~ModuleFactory()
    {
      for (const auto& mod: _stateless_storage)
        delete mod.second;

      for (const auto& mod: _storage)
        delete mod;
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Module<MatFwd>*
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::build(th::Class* obj)
    {
      std::string name = obj->get_classname();
      auto data = dynamic_cast<th::Table*>(obj->get_data());

      auto custom_name = th::get_field<th::String*>(data, "name");

      if (!custom_name)
      {
        auto it = _stateless_storage.find(name);

        if (it != _stateless_storage.end())
          return it->second;
      }

      Module<MatFwd>* mod = nullptr;

      if (name == "nn.Linear")
        mod = new Linear<MatFwd, MatIn, ModelT>(data);
      else if (name == "nn.LookupTable")
        mod =  new LookupTable<MatFwd, MatEmb, ModelT>(data);
      else if (name == "nn.CAddTable")
        mod = new CAddTable<MatFwd>();
      else if (name == "nn.CMulTable")
        mod = new CMulTable<MatFwd>();
      else if (name == "nn.Sigmoid")
        mod = new Sigmoid<MatFwd>();
      else if (name == "nn.Tanh")
        mod = new Tanh<MatFwd>();
      else if (name == "nn.SplitTable")
        mod = new SplitTable<MatFwd>();
      else if (name == "nn.JoinTable")
        mod = new JoinTable<MatFwd>();
      else if (name == "nn.SelectTable")
        mod = new SelectTable<MatFwd>(data);
      else if (name == "nn.Reshape")
        mod = new Reshape<MatFwd>();
      else if (name == "nn.Replicate")
        mod = new Replicate<MatFwd>(data);
      else if (name == "nn.SoftMax")
        mod = new SoftMax<MatFwd>();
      else if (name == "nn.LogSoftMax")
        mod = new LogSoftMax<MatFwd>();
      else if (name == "nn.MM")
        mod = new MM<MatFwd>(data);
      else if (name == "nn.Sum")
        mod = new Sum<MatFwd>(data);
      else if (name == "nn.Sequential")
        mod = new Sequential<MatFwd, MatIn, MatEmb, ModelT>(data, *this);
      else if (name == "nn.ConcatTable")
        mod = new ConcatTable<MatFwd, MatIn, MatEmb, ModelT>(data, *this);
      else if (name == "nn.ParallelTable")
        mod = new ParallelTable<MatFwd, MatIn, MatEmb, ModelT>(data, *this);
      else if (name == "nn.Identity" || name == "nn.Dropout")
        mod = new Identity<MatFwd>();
      else if (name == "nn.gModule")
        mod = new Graph<MatFwd, MatIn, MatEmb, ModelT>(obj, name, *this);
      else
      {
        auto net = th::get_field<th::Class*>(data, "net");

        if (net)
          return build(net);
        else
          throw std::runtime_error(name + " is not supported yet");
      }

      if (custom_name)
        mod->set_custom_name(custom_name->get_value());

      _storage.push_back(mod);

      return mod;
    }

  }
}
