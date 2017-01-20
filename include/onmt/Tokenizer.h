#pragma once

#include <vector>
#include <string>

namespace onmt
{

  class Tokenizer
  {
  public:
    enum class Mode
    {
      Conservative,
      Aggressive
    };

    static const std::string joiner_marker;
    static const std::string feature_marker;

    Tokenizer(Mode mode = Mode::Conservative,
              bool case_feature = false,
              bool joiner_annotate = false,
              bool joiner_new = false,
              const std::string& joiner = joiner_marker);

    std::vector<std::string> tokenize(const std::string& text);

  private:
    Mode _mode;
    bool _case_feature;
    bool _joiner_annotate;
    bool _joiner_new;
    std::string _joiner;

    std::vector<std::string> tokenize_line(const std::string& text);
  };

}
