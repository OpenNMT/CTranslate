#pragma once

#include <vector>
#include <string>

namespace onmt
{

  class ITokenizer
  {
  public:
    static const std::string feature_marker;

    virtual ~ITokenizer() = default;

    virtual void tokenize(const std::string& text,
                          std::vector<std::string>& words,
                          std::vector<std::vector<std::string> >& features) = 0;
    virtual std::string detokenize(const std::vector<std::string>& words,
                                   const std::vector<std::vector<std::string> >& features) = 0;

    // Tokenize/detokenize space-separated tokens.
    virtual std::string tokenize(const std::string& text);
    virtual std::string detokenize(const std::string& text);
  };
}
