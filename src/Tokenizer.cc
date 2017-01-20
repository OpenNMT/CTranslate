#include "onmt/Tokenizer.h"

#include <unicode.h>

#include "onmt/CaseModifier.h"

namespace onmt
{

  const std::string Tokenizer::joiner_marker("￭");
  const std::string Tokenizer::feature_marker("￨");

  Tokenizer::Tokenizer(Mode mode,
                       bool case_feature,
                       bool joiner_annotate,
                       bool joiner_new,
                       const std::string& joiner)
    : _mode(mode)
    , _case_feature(case_feature)
    , _joiner_annotate(joiner_annotate)
    , _joiner_new(joiner_new)
    , _joiner(joiner)
  {
  }

  std::vector<std::string> Tokenizer::tokenize(const std::string& text)
  {
    auto tokens = tokenize_line(text);

    if (_case_feature)
    {
      for (size_t i = 0; i < tokens.size(); ++i)
      {
        std::string new_token;
        char feat = CaseModifier::extract_case(tokens[i], new_token);
        tokens[i] = new_token + feature_marker + feat;
      }
    }

    return tokens;
  }

  std::vector<std::string> Tokenizer::tokenize_line(const std::string& text)
  {
    std::vector<std::string> chars;
    std::vector<unicode_code_point_t> code_points;

    split_utf8(text, chars, code_points);

    std::vector<std::string> tokens;
    std::string token;

    bool letter = false;
    bool number = false;
    bool other = false;
    bool space = true;

    _type_letter type_letter;

    for (size_t i = 0; i < chars.size(); ++i)
    {
      const std::string& c = chars[i];
      unicode_code_point_t v = code_points[i];
      unicode_code_point_t next_v = i + 1 < code_points.size() ? code_points[i + 1] : 0;

      if (is_separator(v))
      {
        if (!space)
        {
          tokens.push_back(token);
          token.clear();
        }

        if (v == 0x200D && _joiner_annotate)
        {
          if (_joiner_new && tokens.size() > 0)
            tokens.push_back(_joiner);
          else
          {
            if (other || (number && is_letter(next_v, type_letter)))
              tokens.back() += _joiner;
            else
              token = _joiner;
          }
        }

        letter = false;
        number = false;
        other = false;
        space = true;
      }
      else
      {
        bool cur_letter = false;
        bool cur_number = false;

        if (v > 32 and v != 0xFEFF)
        {
          cur_letter = is_letter(v, type_letter);
          cur_number = is_number(v);

          if (_mode == Mode::Conservative)
          {
            if (is_number(v)
                || (c == "-" && letter)
                || c == "_"
                || (letter || ((c == "." || c == ",")
                               && (is_number(next_v) || is_letter(next_v, type_letter)))))
              cur_letter = true;
          }
        }

        if (cur_letter)
        {
          if (!letter && !space)
          {
            if (_joiner_annotate && !_joiner_new)
              token += _joiner;
            tokens.push_back(token);
            if (_joiner_annotate && _joiner_new)
              tokens.push_back(_joiner);
            token.clear();
          }
          else if (other && _joiner_annotate && token.empty())
          {
            if (_joiner_new)
              tokens.push_back(_joiner);
            else
              tokens.back() += _joiner;
          }

          token += c;
          letter = true;
          number = false;
          other = false;
          space = false;
        }
        else if (cur_number)
        {
          if (!letter && !space)
          {
            if (_joiner_annotate && !_joiner_new && !letter)
              token += _joiner;
            tokens.push_back(token);
            if (_joiner_annotate && _joiner_new)
              tokens.push_back(_joiner);
            token.clear();
            if (_joiner_annotate && !_joiner_new && letter)
              token += _joiner;
          }
          else if (other && _joiner_annotate)
          {
            if (_joiner_new)
              tokens.push_back(_joiner);
            else
              token = _joiner;
          }

          token += c;
          letter = false;
          number = true;
          other = false;
          space = false;
        }
        else
        {
          if (!space)
          {
            if (_joiner_annotate && !_joiner_new)
              tokens.push_back(_joiner + token);
            else
              tokens.push_back(token);
            if (_joiner_annotate && _joiner_new)
              tokens.push_back(_joiner);
            token.clear();
          }
          else if (other && _joiner_annotate)
          {
            if (_joiner_new)
              tokens.push_back(_joiner);
            else
              token = _joiner;
          }

          token += c;
          tokens.push_back(token);
          token.clear();
          letter = false;
          number = false;
          other = true;
          space = true;
        }
      }
    }

    if (!token.empty())
      tokens.push_back(token);

    return tokens;
  }

}
