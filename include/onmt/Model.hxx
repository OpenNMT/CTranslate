#pragma once

#include "onmt/th/Obj.h"

namespace onmt
{

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  Model<MatFwd, MatIn, MatEmb, ModelT>::Model(const std::string& filename, Profiler& profiler, bool cuda)
    : _module_factory(profiler, cuda)
  {
    THFile* tf = THDiskFile_new(filename.c_str(), "r", 0);
    THFile_binary(tf);
    THDiskFile_longSize(tf, th::dfLongSize);

    th::Obj* obj = read_obj(tf, _env);

    THFile_free(tf);

    th::Table* main = dynamic_cast<th::Table*>(obj);

    load_options(th::get_field<th::Table*>(main, "options"));
    load_dictionaries(th::get_field<th::Table*>(main, "dicts"));
    load_networks(th::get_field<th::Table*>(main, "models"));
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
  void Model<MatFwd, MatIn, MatEmb, ModelT>::load_networks(th::Table* obj,
                                                           std::vector<nn::Module<MatFwd>*>& modules)
  {
    auto modules_set = th::get_field<th::Table*>(obj, "modules");
    auto modules_data = modules_set->get_array();

    for (auto module: modules_data)
    {
      th::Class* mod = dynamic_cast<th::Class*>(module);

      if (mod)
        modules.push_back(_module_factory.build(mod));
      else if (dynamic_cast<th::Table*>(module))
        load_networks(dynamic_cast<th::Table*>(module), modules);
    }
  }

  template <typename MF>
  static void* mark_block(nn::Module<MF>* M, void* t)
  {
    M->set_block((const char*)t);
    return 0;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void Model<MatFwd, MatIn, MatEmb, ModelT>::load_networks(th::Table* obj)
  {
    load_networks(th::get_field<th::Table*>(obj, "encoder"), _encoder_modules);
    _encoder_modules[0]->apply(mark_block<MatFwd>, (void*)"encoder_fwd");
    if (_encoder_modules[1])
      _encoder_modules[1]->apply(mark_block<MatFwd>, (void*)"encoder_bwd");
    load_networks(th::get_field<th::Table*>(obj, "decoder"), _decoder_modules);
    _decoder_modules[0]->apply(mark_block<MatFwd>, (void*)"decoder");
    _decoder_modules[1]->apply(mark_block<MatFwd>, (void*)"generator");
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  nn::Module<MatFwd>* Model<MatFwd, MatIn, MatEmb, ModelT>::get_module(size_t index,
                                                                       std::vector<nn::Module<MatFwd>*>& modules)
  {
    if (index < modules.size())
      return modules[index];
    return nullptr;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  nn::Module<MatFwd>* Model<MatFwd, MatIn, MatEmb, ModelT>::get_encoder_module(size_t index)
  {
    return get_module(index, _encoder_modules);
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  nn::Module<MatFwd>* Model<MatFwd, MatIn, MatEmb, ModelT>::get_decoder_module(size_t index)
  {
    return get_module(index, _decoder_modules);
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
