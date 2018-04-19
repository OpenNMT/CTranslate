#include "onmt/VocabMapping.h"
#include <fstream>

namespace onmt {
  static void _split(std::vector<std::string>& cont, const std::string &str, char delim)
  {
      cont.clear();
      std::size_t current, previous = 0;
      current = str.find(delim);
      while (current != std::string::npos) {
          cont.push_back(str.substr(previous, current - previous));
          previous = current + 1;
          current = str.find(delim, previous);
      }
      cont.push_back(str.substr(previous, current - previous));
  }

	VocabMapping::VocabMapping(const std::string &map_file, const Dictionary &dict) {
    std::ifstream mapf(map_file);
    if (!mapf.is_open())
      throw std::invalid_argument("Unable to open dictionary vocab mapping file `" + map_file + "`");
    std::string line;
    while (std::getline(mapf, line)) {
      std::vector<std::string> parts;
      _split(parts, line, '\t');
      if (parts.size() > 1) {
        std::vector<std::string> wparts;
        _split(wparts, parts[0], ' ');
        size_t l = wparts.size();
        for(size_t i = 1; i < parts.size(); i++) {
          _split(wparts, parts[i], ' ');
          for(auto tok: wparts) {
            _map_rules[l].insert(std::make_pair(parts[0], dict.get_word_id(tok)));
          }
        }
      }
    }
	}
	std::set<size_t> VocabMapping::build_subdict(const std::vector<std::string> &words) {
		std::set<size_t> r;
    std::multimap<std::string, size_t>::const_iterator it;

    /* empty source string */
    it = _map_rules[0].find("");
    while (it != _map_rules[0].end()) {
      r.insert(it->second);
      it++;
    }

    for(size_t i=0; i<words.size(); i++) {
      std::string tok = words[i];
      size_t h = 0;
      do {
        if (h > 0) {
          if (i+h >= words.size()) break;
          tok += " " + words[i+h];
        }
        it = _map_rules[h+1].find(tok);
        while (it != _map_rules[h+1].end() && it->first == tok) {
          r.insert(it->second);
          it++;
        }
        h++;
      } while (h+1 < _map_rules.size());
    }
		return r;
	}
}