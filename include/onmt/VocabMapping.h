#pragma once

#include <set>
#include <map>
#include <string>
#include <vector>
#include "onmt/Dictionary.h"

namespace onmt {
  class VocabMapping {
  public:
    VocabMapping(const std::string &map_file, const Dictionary &dict);
    std::set<size_t> build_subdict(const std::vector<std::string> &words);
  private:
    std::vector< std::multimap<std::string, size_t> > _map_rules;
  };
}