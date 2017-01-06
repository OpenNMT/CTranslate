#include "onmt/Dictionary.h"

#ifdef ANDROID_GNUSTL_COMPAT
#  include "onmt/android_gnustl_compat.h"
#endif

#include <fstream>

#include "onmt/th/Utils.h"

namespace onmt
{

  const size_t Dictionary::pad_id = 0;
  const size_t Dictionary::unk_id = 1;
  const size_t Dictionary::bos_id = 2;
  const size_t Dictionary::eos_id = 3;

  Dictionary::Dictionary()
  {
  }

  Dictionary::Dictionary(th::Class* dict)
  {
    load(dict);
  }

  void Dictionary::load(th::Class* dict)
  {
    auto dict_data = dynamic_cast<th::Table*>(dict->get_data());
    auto id2word = th::get_field<th::Table*>(dict_data, "idxToLabel");

    auto array = id2word->get_array();

    for (size_t i = 0; i < array.size(); ++i)
    {
      const std::string& word = dynamic_cast<th::String*>(array[i])->get_value();
      _id2word.push_back(word);
      _word2id[word] = i;
    }
  }

  size_t Dictionary::get_size() const
  {
    return _id2word.size();
  }

  size_t Dictionary::get_word_id(const std::string& word) const
  {
    auto it = _word2id.find(word);

    if (it == _word2id.cend())
      return unk_id;

    return it->second;
  }

  const std::string& Dictionary::get_id_word(size_t id) const
  {
    return _id2word[id];
  }

}
