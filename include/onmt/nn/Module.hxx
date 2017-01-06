#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Module<MatFwd>::Module(const std::string& name)
      : _name(name)
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> Module<MatFwd>::forward(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> out;
      out.push_back(forward(input[0]));
      return wrap_return(out);
    }

    template <typename MatFwd>
    MatFwd Module<MatFwd>::forward(MatFwd& input) const
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
    std::vector<MatFwd>& Module<MatFwd>::wrap_return(std::vector<MatFwd>& output) const
    {
      if (_post_process)
        _post_process(output);

      return output;
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

  }
}
