#pragma once

#include <memory>
#include <string>
#include <vector>

#include "onmt/Profiler.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Module
    {
    public:
      Module(const std::string& name)
        : _name(name)
        , _profile(true)
        , _profiler(nullptr)
        , _outputs(1)
        , _output(_outputs.front())
      {
      }

      Module(const std::string& name, bool profile)
        : _name(name)
        , _profile(profile)
        , _profiler(nullptr)
        , _outputs(1)
        , _output(_outputs.front())
      {
      }

      virtual ~Module()
      {
      }

      const std::vector<MatFwd>& forward(const std::vector<MatFwd>& inputs)
      {
        if (_profile && _profiler)
          _profiler->start();

        forward_impl(inputs);

        if (_profile && _profiler)
          _profiler->stop(_block + (!_custom_name.empty() ? _custom_name : _name));

        if (_post_process)
          _post_process(_outputs);

        return _outputs;
      }

      const MatFwd& forward_one(const MatFwd& input)
      {
        return forward(std::vector<MatFwd>(1, input))[0];
      }

      virtual void forward_impl(const std::vector<MatFwd>& inputs)
      {
        forward_impl(inputs.front());
      }

      virtual void forward_impl(const MatFwd& input)
      {
        _output = input;
      }

      virtual Module<MatFwd>* find(const std::string& custom_name)
      {
        if (_custom_name == custom_name)
          return this;

        return nullptr;
      }

      virtual void* apply(void* (*func)(Module<MatFwd>*, void*), void* data)
      {
        return func(this, data);
      }

      std::function<void(std::vector<MatFwd>&)>& post_process_fun()
      {
        return _post_process;
      }

      const std::vector<MatFwd>& get_outputs() const
      {
        return _outputs;
      }

      const std::string& get_name() const
      {
        return _name;
      }

      const std::string& get_custom_name() const
      {
        return _custom_name;
      }

      virtual std::string get_details() const
      {
        return "";
      }

      void set_custom_name(const std::string& custom_name)
      {
        _custom_name = custom_name;
      }

      void set_profiler(Profiler& profiler)
      {
        _profiler = &profiler;
      }

      void set_block(const char* s)
      {
        _block = std::string(s) + ":";
      }

    protected:
      std::string _name;
      std::string _custom_name;
      bool _profile;
      std::string _block;
      Profiler* _profiler;

      std::vector<MatFwd> _outputs;
      MatFwd& _output;

      std::function<void(std::vector<MatFwd>&)> _post_process;
    };

  }
}
