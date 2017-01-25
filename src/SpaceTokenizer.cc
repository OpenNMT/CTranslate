#include "onmt/SpaceTokenizer.h"

#include <sstream>

#include "onmt/unicode/Unicode.h"

namespace onmt
{

  ITokenizer& SpaceTokenizer::get_instance()
  {
    static SpaceTokenizer tokenizer;
    return tokenizer;
  }

  void SpaceTokenizer::tokenize(const std::string& text,
                                std::vector<std::string>& words,
                                std::vector<std::vector<std::string> >& features)
  {
    std::vector<std::string> chunks = unicode::split_utf8(text, " ");

    for (const auto& chunk: chunks)
    {
      size_t i = 0;
      int sep_offset = -ITokenizer::feature_marker.length();

      do {
        int start = sep_offset + ITokenizer::feature_marker.length();
        sep_offset = chunk.find(ITokenizer::feature_marker, start);
        std::string sub = chunk.substr(start, sep_offset);

        if (i == 0)
          words.push_back(sub);
        else
        {
          if (features.size() < i)
            features.emplace_back(1, sub);
          else
            features[i-1].push_back(sub);
        }

        i++;
      } while (static_cast<size_t>(sep_offset) != std::string::npos);
    }
  }

  std::string SpaceTokenizer::detokenize(const std::vector<std::string>& words,
                                         const std::vector<std::vector<std::string> >& features)
  {
    std::ostringstream oss;

    for (size_t i = 0; i < words.size(); ++i)
    {
      if (i > 0)
        oss << " ";
      oss << words[i];

      if (!features.empty())
      {
        for (size_t j = 0; j < features.size(); ++j)
          oss << ITokenizer::feature_marker << features[j][i];
      }
    }

    return oss.str();
  }

}
