#pragma once

#include <unordered_map>

#include "onmt/Dictionary.h"
#include "onmt/nn/ModuleFactory.h"
#include "onmt/th/Env.h"

namespace onmt
{

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  class Model
  {
  public:
    Model(const std::string& filename);

    void create_graph(nn::ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory,
                      std::vector<size_t>& encoder,
                      std::vector<size_t>& decoder);

    const Dictionary& get_src_dict() const;
    const Dictionary& get_tgt_dict() const;
    const std::vector<Dictionary>& get_src_feat_dicts() const;
    const std::vector<Dictionary>& get_tgt_feat_dicts() const;

    const std::string& get_option_string(const std::string& key) const;
    bool get_option_flag(const std::string& key, bool default_value = false) const;

    template <typename T = double>
    T get_option_value(const std::string& key, T default_value = 0) const;

  private:
    void load_options(th::Table* obj);
    void load_dictionaries(th::Table* obj, Dictionary& words, std::vector<Dictionary>& features);
    void load_dictionaries(th::Table* obj);

    void load_modules(th::Table* obj,
                      std::vector<size_t>& modules,
                      nn::ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& module_factory) const;

    th::Env _env;
    th::Table* _root;

    Dictionary _src_dict;
    Dictionary _tgt_dict;
    std::vector<Dictionary> _src_feat_dicts;
    std::vector<Dictionary> _tgt_feat_dicts;

    std::unordered_map<std::string, double> _options_value;
    std::unordered_map<std::string, std::string> _options_str;
    const std::string _empty_str;
  };

}

#include "Model.hxx"
