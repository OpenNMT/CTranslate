#pragma once

#include <unordered_map>

#include "onmt/Dictionary.h"
#include "onmt/nn/Module.h"
#include "onmt/th/Env.h"

namespace onmt
{

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  class Model
  {
  public:
    Model(const std::string& filename);
    ~Model();

    nn::Module<MatFwd>* get_encoder_module(size_t index);
    nn::Module<MatFwd>* get_decoder_module(size_t index);

    const Dictionary& get_src_dict() const;
    const Dictionary& get_tgt_dict() const;
    const std::vector<Dictionary>& get_src_feat_dicts() const;
    const std::vector<Dictionary>& get_tgt_feat_dicts() const;

    const std::string& get_option_string(const std::string& key) const;
    bool get_option_flag(const std::string& key, bool default_value = false) const;

    template <typename T = double>
    T get_option_value(const std::string& key, T default_value = 0) const;

  private:
    nn::Module<MatFwd>* get_module(size_t index, std::vector<nn::Module<MatFwd>*>& modules);

    void load_options(th::Table* obj);
    void load_dictionaries(th::Table* obj, Dictionary& words, std::vector<Dictionary>& features);
    void load_dictionaries(th::Table* obj);
    void load_networks(th::Table* obj, std::vector<nn::Module<MatFwd>*>& modules);
    void load_networks(th::Table* obj);

    th::Env _env;

    std::vector<nn::Module<MatFwd>*> _encoder_modules;
    std::vector<nn::Module<MatFwd>*> _decoder_modules;

    Dictionary _src_dict;
    Dictionary _tgt_dict;
    std::vector<Dictionary> _src_feat_dicts;
    std::vector<Dictionary> _tgt_feat_dicts;

    std::unordered_map<std::string, double> _options_value;
    std::unordered_map<std::string, std::string> _options_str;
    std::string _empty_str = "";
  };

}

#include "Model.hxx"
