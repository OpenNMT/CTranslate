#pragma once

#include "onmt/th/Obj.h"
#include "onmt/Utils.h"

namespace onmt
{

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  Model<MatFwd, MatIn, MatEmb, ModelT>::Model(const std::string& filename)
  {
    ONMT_LOG_STREAM_SEV("Loading '" << filename << "'...", boost::log::trivial::info);
    THFile* tf = THDiskFile_new(filename.c_str(), "r", 0);
    THFile_binary(tf);
    THDiskFile_longSize(tf, th::dfLongSize);

    th::Obj* obj = read_obj(tf, _env);

    THFile_free(tf);

    _root = dynamic_cast<th::Table*>(obj);

    load_options(th::get_field<th::Table*>(_root, "options"));
    load_dictionaries(th::get_field<th::Table*>(_root, "dicts"));
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void Model<MatFwd, MatIn, MatEmb, ModelT>::load_options(th::Table* obj)
  {
    const auto& opt = obj->get_object();

    for (auto pair: opt)
    {
      const std::string& key = pair.first;

      if (dynamic_cast<th::Number*>(pair.second))
      {
        double value = static_cast<double>(dynamic_cast<th::Number*>(pair.second)->get_value());
        _options_value[key] = value;
      }
      else if (dynamic_cast<th::Boolean*>(pair.second))
      {
        bool value = dynamic_cast<th::Boolean*>(pair.second)->get_value();
        _options_value[key] = value ? 1 : 0;
      }
      else if (dynamic_cast<th::String*>(pair.second))
      {
        const std::string& str = dynamic_cast<th::String*>(pair.second)->get_value();
        _options_str[key] = str;
      }
    }
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void Model<MatFwd, MatIn, MatEmb, ModelT>::load_dictionaries(th::Table* obj,
                                                               Dictionary& words,
                                                               std::vector<Dictionary>& features)
  {
    words.load(th::get_field<th::Class*>(obj, "words"));

    auto features_set = th::get_field<th::Table*>(obj, "features");
    auto features_dicts = features_set->get_array();

    for (size_t i = 0; i <  features_dicts.size(); ++i)
    {
      features.emplace_back(dynamic_cast<th::Class*>(features_dicts[i]));
    }
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void Model<MatFwd, MatIn, MatEmb, ModelT>::load_dictionaries(th::Table* obj)
  {
    load_dictionaries(th::get_field<th::Table*>(obj, "src"), _src_dict, _src_feat_dicts);
    load_dictionaries(th::get_field<th::Table*>(obj, "tgt"), _tgt_dict, _tgt_feat_dicts);
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void Model<MatFwd, MatIn, MatEmb, ModelT>::load_modules(
    th::Table* obj,
    std::vector<size_t>& modules,
    nn::ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& module_factory) const
  {
    auto modules_set = th::get_field<th::Table*>(obj, "modules");
    auto modules_data = modules_set->get_array();

    for (auto module: modules_data)
    {
      th::Class* mod = dynamic_cast<th::Class*>(module);

      if (mod)
        modules.push_back(module_factory.build(mod));
      else if (dynamic_cast<th::Table*>(module))
        load_modules(dynamic_cast<th::Table*>(module), modules, module_factory);
    }
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  struct record_graph {
    record_graph():currentGraph(nullptr),attentionGraph(nullptr) {}
    nn::Module<MatFwd, MatIn, MatEmb, ModelT> *currentGraph;
    nn::Module<MatFwd, MatIn, MatEmb, ModelT> *attentionGraph;
  };

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  static void* _find_attentiongraph(nn::Module<MatFwd, MatIn, MatEmb, ModelT>* M, void* t)
  {
    if (M->get_name() == "nn.gModule"){
      ((record_graph<MatFwd, MatIn, MatEmb, ModelT>*)t)->currentGraph = M;
    }
    else if (M->get_custom_name() == "softmaxAttn") {
      ((record_graph<MatFwd, MatIn, MatEmb, ModelT>*)t)->attentionGraph =
        ((record_graph<MatFwd, MatIn, MatEmb, ModelT>*)t)->currentGraph;
    }
    return 0;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  static void* _mark_block(nn::Module<MatFwd, MatIn, MatEmb, ModelT>* M, void* t)
  {
    M->set_block((const char*)t);
    return 0;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void Model<MatFwd, MatIn, MatEmb, ModelT>::create_graph(
    nn::ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory,
    std::vector<size_t>& encoder,
    std::vector<size_t>& decoder)
  {
    auto models = th::get_field<th::Table*>(_root, "models");
    load_modules(th::get_field<th::Table*>(models, "encoder"), encoder, factory);
    load_modules(th::get_field<th::Table*>(models, "decoder"), decoder, factory);

    /* annotate the different modules for profiling */
    factory.get_module(encoder[0])->apply(_mark_block<MatFwd, MatIn, MatEmb, ModelT>, (void*)"encoder_fwd");
    if (encoder.size() > 1)
      factory.get_module(encoder[1])->apply(_mark_block<MatFwd, MatIn, MatEmb, ModelT>, (void*)"encoder_bwd");
    auto decoder_mod = factory.get_module(decoder[0]);
    decoder_mod->apply(_mark_block<MatFwd, MatIn, MatEmb, ModelT>, (void*)"decoder");
    factory.get_module(decoder[1])->apply(_mark_block<MatFwd, MatIn, MatEmb, ModelT>, (void*)"generator");

    /* find the attention module and annotate it specifically */
    record_graph<MatFwd, MatIn, MatEmb, ModelT> rg;
    decoder_mod->apply(_find_attentiongraph<MatFwd, MatIn, MatEmb, ModelT>, (void*)&rg);
    if (rg.attentionGraph)
      rg.attentionGraph->apply(_mark_block<MatFwd, MatIn, MatEmb, ModelT>, (void*)"attention");
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  const Dictionary& Model<MatFwd, MatIn, MatEmb, ModelT>::get_src_dict() const
  {
    return _src_dict;
  }
  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  const Dictionary& Model<MatFwd, MatIn, MatEmb, ModelT>::get_tgt_dict() const
  {
    return _tgt_dict;
  }
  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  const std::vector<Dictionary>& Model<MatFwd, MatIn, MatEmb, ModelT>::get_src_feat_dicts() const
  {
    return _src_feat_dicts;
  }
  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  const std::vector<Dictionary>& Model<MatFwd, MatIn, MatEmb, ModelT>::get_tgt_feat_dicts() const
  {
    return _tgt_feat_dicts;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  const std::string& Model<MatFwd, MatIn, MatEmb, ModelT>::get_option_string(const std::string& key) const
  {
    auto it = _options_str.find(key);

    if (it == _options_str.cend())
      return _empty_str;

    return it->second;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  bool Model<MatFwd, MatIn, MatEmb, ModelT>::get_option_flag(const std::string& key,
                                                             bool default_value) const
  {
    return get_option_value<int>(key, static_cast<int>(default_value)) == 1;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  template <typename T>
  T Model<MatFwd, MatIn, MatEmb, ModelT>::get_option_value(const std::string& key, T default_value) const
  {
    auto it = _options_value.find(key);

    if (it == _options_value.cend())
      return default_value;

    return static_cast<T>(it->second);
  }

}
