#include "onmt/SubDict.h"

#include <fstream>

namespace onmt
{

  static void _split(std::vector<std::string>& cont, const std::string& str, char delim)
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

  SubDict::SubDict(const std::string& map_file, const Dictionary& dict)
  {
    if (!map_file.empty())
    {
      std::ifstream mapf(map_file);
      if (!mapf.is_open())
        throw std::invalid_argument("Unable to open dictionary vocab mapping file `" + map_file + "`");
      _map_rules.resize(1);
      _map_rules[0].insert(std::make_pair("", Dictionary::unk_id));
      _map_rules[0].insert(std::make_pair("", Dictionary::bos_id));
      _map_rules[0].insert(std::make_pair("", Dictionary::eos_id));
      _map_rules[0].insert(std::make_pair("", Dictionary::pad_id));
      std::string line;
      while (std::getline(mapf, line))
      {
        std::vector<std::string> parts;
        _split(parts, line, '\t');
        if (parts.size() > 1) {
          std::vector<std::string> wparts;
          size_t l = 0;
          if (parts[0].length())
          {
            _split(wparts, parts[0], ' ');
            l = wparts.size();
          }
          if (l >= _map_rules.size())
            _map_rules.resize(l + 1);
          for (size_t i = 1; i < parts.size(); i++)
          {
            _split(wparts, parts[i], ' ');
            for (auto tok: wparts)
              _map_rules[l].insert(std::make_pair(parts[0], dict.get_word_id(tok)));
          }
        }
      }
    }
  }

  void SubDict::extract(const std::vector<std::string>& words, std::set<size_t>& r)
  {
    std::multimap<std::string, size_t>::const_iterator it;

    /* empty source string */
    it = _map_rules[0].find("");
    while (it != _map_rules[0].end())
    {
      r.insert(it->second);
      it++;
    }

    for (size_t i = 0; i < words.size(); i++)
    {
      std::string tok = words[i];
      size_t h = 0;
      do {
        if (h > 0)
        {
          if (i + h >= words.size())
            break;
          tok += " " + words[i + h];
        }
        it = _map_rules[h + 1].find(tok);
        while (it != _map_rules[h + 1].end() && it->first == tok)
        {
          r.insert(it->second);
          it++;
        }
        h++;
      } while (h + 1 < _map_rules.size());
    }
  }

  void SubDict::reduce_linearweight(
      const Eigen::Map<const Eigen::RowMajorMat<float> >& w,
      const Eigen::Map<const Eigen::RowMajorMat<float> >& b,
      Eigen::RowMajorMat<float>& rw,
      Eigen::RowMajorMat<float>& rb,
      const std::vector<size_t>& v)
  {
    rw.resize(v.size(), w.cols());
    rb.resize(v.size(), 1);
    /* build sub-matrix where the number of rows is restricted to rows in row_subset */
    for (size_t i = 0; i < v.size(); i++) {
      rw.row(i) = w.row(v[i]);
      rb.row(i) = b.row(v[i]);
    }
  }

}
