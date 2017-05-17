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
      {
      }

      Module(const std::string& name, bool profile)
        : _name(name)
        , _profile(profile)
        , _profiler(nullptr)
      {
      }

      virtual ~Module()
      {
      }

      std::vector<MatFwd> forward(std::vector<MatFwd>& input)
      {
        if (_profile && _profiler)
          _profiler->start();

        auto output = forward_impl(input);

        if (_profile && _profiler)
          _profiler->stop(!_custom_name.empty() ? _custom_name : _name);

        if (_post_process)
          _post_process(output);

        return output;
      }

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input)
      {
        return std::vector<MatFwd>(1, forward_impl(input[0]));
      }

      virtual MatFwd forward_impl(MatFwd& input)
      {
        return input;
      }

      virtual Module<MatFwd>* find(const std::string& custom_name)
      {
        if (_custom_name == custom_name)
          return this;

        return nullptr;
      }

      std::function<void(std::vector<MatFwd>&)>& post_process_fun()
      {
        return _post_process;
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

    protected:
      std::string _name;
      std::string _custom_name;
      bool _profile;
      Profiler* _profiler;

      std::function<void(std::vector<MatFwd>&)> _post_process;
    };

  }
}
