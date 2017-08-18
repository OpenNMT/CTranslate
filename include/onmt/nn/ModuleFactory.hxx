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
#include "onmt/nn/Squeeze.h"
#include "onmt/nn/MulConstant.h"

#include "onmt/nn/Sequential.h"
#include "onmt/nn/ParallelTable.h"
#include "onmt/nn/ConcatTable.h"

#include "onmt/nn/Graph.h"

#ifdef WITH_CUDA
#  include "onmt/nn/cuLinear.h"
#endif

namespace onmt
{
  namespace nn
  {


    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::ModuleFactory(Profiler& profiler, bool cuda)
      : _profiler(profiler)
      , _cuda(cuda)
    {
      if (_cuda)
      {
#ifdef WITH_CUDA
        CUBLAS_CHECK(cublasCreate(&_handle));
#else
        throw std::runtime_error("CTranslate was not compiled with CUDA support");
#endif
      }
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::~ModuleFactory()
    {
#ifdef WITH_CUDA
      if (_cuda)
        CUBLAS_CHECK(cublasDestroy(_handle));
#endif

      for (const auto& mod: _storage)
        delete mod;
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Module<MatFwd>*
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::build(th::Class* obj)
    {
      std::string name = obj->get_classname();
      auto data = dynamic_cast<th::Table*>(obj->get_data());

      Module<MatFwd>* mod = nullptr;

      if (name == "nn.Linear")
      {
#ifdef WITH_CUDA
        if (_cuda)
          mod = new cuLinear<MatFwd, MatIn, ModelT>(data, _handle);
        else
#endif
          mod = new Linear<MatFwd, MatIn, ModelT>(data);
      }
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
        mod = new Reshape<MatFwd>(data);
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
      else if (name == "nn.Squeeze")
        mod = new Squeeze<MatFwd>(data);
      else if (name == "nn.MulConstant")
        mod = new MulConstant<MatFwd, ModelT>(data);
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

      auto custom_name = th::get_field<th::String*>(data, "name");

      if (custom_name)
        mod->set_custom_name(custom_name->get_value());

      mod->set_profiler(_profiler);

      _storage.push_back(mod);

      return mod;
    }

  }
}
