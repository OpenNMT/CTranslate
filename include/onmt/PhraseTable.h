#pragma once

#include <string>
#include <unordered_map>

namespace onmt
{

  class PhraseTable
  {
  public:
    PhraseTable(const std::string& file);

    bool is_empty() const;
    size_t get_size() const;

    std::string lookup(const std::string& src) const;

  private:
    std::unordered_map<std::string, std::string> _src_to_tgt;
  };

}
