#include "onmt/SubDict.h"

#include <fstream>

namespace onmt
{

  SubDict::SubDict(const std::string& map_file, const Dictionary& dict)
  {
    if (!map_file.empty())
    {
      std::ifstream mapf(map_file);
      if (!mapf.is_open())
        throw std::invalid_argument("Unable to open dictionary vocab mapping file `" + map_file + "`");
      std::string line;
      while (std::getline(mapf, line))
      {
        std::string token;
        std::string key;
        std::vector<size_t> values;
        bool target = false;
        size_t ngram = 1;

        for (size_t i = 0; i < line.length(); ++i)
        {
          if (line[i] == '\t')
          {
            target = true;
            std::swap(key, token);
          }
          else if (line[i] == ' ')
          {
            if (target)
            {
              values.push_back(dict.get_word_id(token));
              token.clear();
            }
            else
            {
              token += line[i];
              ++ngram;
            }
          }
          else
            token += line[i];
        }

        if (!token.empty())
          values.push_back(dict.get_word_id(token));

        if (ngram > _map_rules.size())
          _map_rules.resize(ngram);

        _map_rules[ngram - 1][key] = values;
      }
    }
  }

  void SubDict::extract(const std::vector<std::string>& words, std::set<size_t>& r) const
  {
    r.insert(Dictionary::unk_id);
    r.insert(Dictionary::bos_id);
    r.insert(Dictionary::eos_id);
    r.insert(Dictionary::pad_id);

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
        auto it = _map_rules[h].find(tok);
        if (it != _map_rules[h].end())
        {
          for (const auto& v: it->second)
            r.insert(v);
        }
        h++;
      } while (h < _map_rules.size());
    }
  }

}
