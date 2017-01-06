#include "onmt/PhraseTable.h"

#include <fstream>

namespace onmt
{

  static const std::string separator = "|||";

  PhraseTable::PhraseTable(const std::string& file)
  {
    if (!file.empty())
    {
      std::ifstream in(file.c_str());
      std::string line;

      while (std::getline(in, line))
      {
        size_t sep_idx = line.find(separator);

        std::string src = line.substr(0, sep_idx);
        std::string tgt = line.substr(sep_idx + separator.length());

        _src_to_tgt[src] = tgt;
      }
    }
  }

  bool PhraseTable::is_empty() const
  {
    return _src_to_tgt.empty();
  }

  size_t PhraseTable::get_size() const
  {
    return _src_to_tgt.size();
  }

  std::string PhraseTable::lookup(const std::string& src) const
  {
    auto it = _src_to_tgt.find(src);

    if (it == _src_to_tgt.cend())
      return "";

    return it->second;
  }

}
