#pragma once

#include "onmt/ITokenizer.h"

namespace onmt
{

  // This Tokenizer implements the behaviour of OpenNMT's tools/tokenize.lua.
  class Tokenizer: public ITokenizer
  {
  public:
    enum class Mode
    {
      Conservative,
      Aggressive
    };

    static const std::string joiner_marker;

    Tokenizer(Mode mode = Mode::Conservative,
              bool case_feature = false,
              bool joiner_annotate = false,
              bool joiner_new = false,
              const std::string& joiner = joiner_marker);
    Tokenizer(bool case_feature = false,
              const std::string& joiner = joiner_marker);

    void tokenize(const std::string& text,
                  std::vector<std::string>& words,
                  std::vector<std::vector<std::string> >& features) override;

    std::string detokenize(const std::vector<std::string>& words,
                           const std::vector<std::vector<std::string> >& features) override;

  private:
    Mode _mode;
    bool _case_feature;
    bool _joiner_annotate;
    bool _joiner_new;
    std::string _joiner;

    bool has_left_join(const std::string& word);
    bool has_right_join(const std::string& word);
  };

}
