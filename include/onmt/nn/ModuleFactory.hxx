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

#ifdef WITH_QLINEAR
#  include "onmt/nn/qLinear.h"
#endif

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::ModuleFactory(Profiler& profiler, bool cuda, bool qlinear)
      : _profiler(&profiler)
      , _cuda(cuda)
      , _qlinear(qlinear)
    {
      if (_cuda)
      {
#ifdef WITH_CUDA
        CUBLAS_CHECK(cublasCreate(&_handle));
#else
        throw std::runtime_error("CTranslate was not compiled with CUDA support");
#endif
      }
      if (_qlinear)
      {
#ifndef WITH_QLINEAR
        throw std::runtime_error("CTranslate was not compiled with QLINEAR support");
#endif
      }
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::ModuleFactory(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& other)
      : _profiler(nullptr)
      , _cuda(other._cuda)
      , _qlinear(other._qlinear)
    {
      if (_cuda)
      {
#ifdef WITH_CUDA
        CUBLAS_CHECK(cublasCreate(&_handle));
#else
        throw std::runtime_error("CTranslate was not compiled with CUDA support");
#endif
      }
      if (_qlinear)
      {
#ifndef WITH_QLINEAR
        throw std::runtime_error("CTranslate was not compiled with QLINEAR support");
#endif
      }

      for (const auto& mod: other._storage)
      {
        _storage.push_back(mod->clone(this));
#ifdef WITH_CUDA
        if (_cuda)
        {
          auto cul = dynamic_cast<cuLinear<MatFwd, MatIn, MatEmb, ModelT>*>(_storage.back());
          if (cul)
            cul->set_handle(&_handle);
        }
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
    size_t
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::build(th::Class* obj)
    {
      std::string name = obj->get_classname();
      auto data = dynamic_cast<th::Table*>(obj->get_data());

      Module<MatFwd, MatIn, MatEmb, ModelT>* mod = nullptr;

      if (name == "nn.Linear")
      {
#ifdef WITH_CUDA
        if (_cuda)
          mod = new cuLinear<MatFwd, MatIn, MatEmb, ModelT>(data, &_handle);
        else
#endif
#ifdef WITH_QLINEAR
        if (_qlinear)
          mod = new qLinear<MatFwd, MatIn, MatEmb, ModelT>(data);
        else
#endif
          mod = new Linear<MatFwd, MatIn, MatEmb, ModelT>(data);
      }
      else if (name == "nn.LookupTable")
        mod =  new LookupTable<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.CAddTable")
        mod = new CAddTable<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.CMulTable")
        mod = new CMulTable<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.Sigmoid")
        mod = new Sigmoid<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.Tanh")
        mod = new Tanh<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.SplitTable")
        mod = new SplitTable<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.JoinTable")
        mod = new JoinTable<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.SelectTable")
        mod = new SelectTable<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.Reshape")
        mod = new Reshape<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.Replicate")
        mod = new Replicate<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.SoftMax")
        mod = new SoftMax<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.LogSoftMax")
        mod = new LogSoftMax<MatFwd, MatIn, MatEmb, ModelT>();
      else if (name == "nn.MM")
        mod = new MM<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.Sum")
        mod = new Sum<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.Squeeze")
        mod = new Squeeze<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.MulConstant")
        mod = new MulConstant<MatFwd, MatIn, MatEmb, ModelT>(data);
      else if (name == "nn.Sequential")
        mod = new Sequential<MatFwd, MatIn, MatEmb, ModelT>(data, *this);
      else if (name == "nn.ConcatTable")
        mod = new ConcatTable<MatFwd, MatIn, MatEmb, ModelT>(data, *this);
      else if (name == "nn.ParallelTable")
        mod = new ParallelTable<MatFwd, MatIn, MatEmb, ModelT>(data, *this);
      else if (name == "nn.Identity" || name == "nn.Dropout")
        mod = new Identity<MatFwd, MatIn, MatEmb, ModelT>();
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

      size_t id = _storage.size();
      _storage.push_back(mod);

      return id;
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Module<MatFwd, MatIn, MatEmb, ModelT>*
    ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::get_module(size_t id) const
    {
      return _storage.at(id);
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    void ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>::set_profiler(Profiler& profiler)
    {
      _profiler = &profiler;
      for (auto& mod : _storage)
        mod->set_profiler(_profiler);
    }

  }
}
