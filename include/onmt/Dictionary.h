#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "onmt/th/Obj.h"

namespace onmt
{

  class Dictionary
  {
  public:
    static const size_t pad_id;
    static const size_t unk_id;
    static const size_t bos_id;
    static const size_t eos_id;

    Dictionary();
    Dictionary(th::Class* dict);

    void load(th::Class* dict);

    size_t get_size() const;

    size_t get_word_id(const std::string& word) const;
    const std::string& get_id_word(size_t id) const;

  private:
    std::vector<std::string> _id2word;
    std::unordered_map<std::string, size_t> _word2id;
  };

}
