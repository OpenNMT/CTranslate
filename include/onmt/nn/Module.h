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
      Module(const std::string& name);
      Module(const std::string& name, bool profile);
      virtual ~Module() {}

      std::vector<MatFwd> forward(std::vector<MatFwd>& input) const;

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const;
      virtual MatFwd forward_impl(MatFwd& input) const;

      virtual Module<MatFwd>* find(const std::string& custom_name);

      std::function<void(std::vector<MatFwd>&)>& post_process_fun();

      const std::string& get_name() const;
      const std::string& get_custom_name() const;
      virtual std::string get_details() const;

      void set_custom_name(const std::string& custom_name);
      void set_profiler(Profiler& profiler);

    protected:
      std::string _name;
      std::string _custom_name;
      bool _profile;
      mutable Profiler* _profiler;

      std::function<void(std::vector<MatFwd>&)> _post_process;
    };

  }
}

#include "onmt/nn/Module.hxx"
