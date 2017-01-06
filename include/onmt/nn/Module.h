#pragma once

#include <memory>
#include <string>
#include <vector>

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Module
    {
    public:
      Module(const std::string& name);
      virtual ~Module() {}

      virtual std::vector<MatFwd> forward(std::vector<MatFwd>& input) const;
      virtual MatFwd forward(MatFwd& input) const;
      virtual Module<MatFwd>* find(const std::string& custom_name);

      std::function<void(std::vector<MatFwd>&)>& post_process_fun();
      std::vector<MatFwd>& wrap_return(std::vector<MatFwd>& output) const;

      const std::string& get_name() const;
      const std::string& get_custom_name() const;
      virtual std::string get_details() const;

      void set_custom_name(const std::string& custom_name);

    protected:
      std::string _name;
      std::string _custom_name;

      std::function<void(std::vector<MatFwd>&)> _post_process;
    };

  }
}

#include "onmt/nn/Module.hxx"
