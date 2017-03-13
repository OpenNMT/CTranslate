#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Module<MatFwd>::Module(const std::string& name)
      : _name(name)
      , _profile(true)
      , _profiler(nullptr)
    {
    }

    template <typename MatFwd>
    Module<MatFwd>::Module(const std::string& name, bool profile)
      : _name(name)
      , _profile(profile)
      , _profiler(nullptr)
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> Module<MatFwd>::forward(std::vector<MatFwd>& input) const
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

    template <typename MatFwd>
    std::vector<MatFwd> Module<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      return std::vector<MatFwd>(1, forward_impl(input[0]));
    }

    template <typename MatFwd>
    MatFwd Module<MatFwd>::forward_impl(MatFwd& input) const
    {
      return input;
    }

    template <typename MatFwd>
    Module<MatFwd>* Module<MatFwd>::find(const std::string& custom_name)
    {
      if (_custom_name == custom_name)
        return this;

      return nullptr;
    }

    template <typename MatFwd>
    std::function<void(std::vector<MatFwd>&)>& Module<MatFwd>::post_process_fun()
    {
      return _post_process;
    }

    template <typename MatFwd>
    const std::string& Module<MatFwd>::get_name() const
    {
      return _name;
    }

    template <typename MatFwd>
    const std::string& Module<MatFwd>::get_custom_name() const
    {
      return _custom_name;
    }

    template <typename MatFwd>
    std::string Module<MatFwd>::get_details() const
    {
      return "";
    }

    template <typename MatFwd>
    void Module<MatFwd>::set_custom_name(const std::string& custom_name)
    {
      _custom_name = custom_name;
    }

    template <typename MatFwd>
    void Module<MatFwd>::set_profiler(Profiler& profiler)
    {
      _profiler = &profiler;
    }

  }
}
