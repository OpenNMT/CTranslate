#include "onmt/Tokenizer.h"

#include <unicode.h>
#include <boost/algorithm/string.hpp>

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

  Tokenizer::Tokenizer(bool case_feature,
                       const std::string& joiner)
    : _mode(Mode::Conservative)
    , _case_feature(case_feature)
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
        auto data = CaseModifier::extract_case(tokens[i]);
        tokens[i] = data.first + feature_marker + data.second;
      }
    }

    return tokens;
  }

  std::string Tokenizer::detokenize(const std::vector<std::string>& tokens)
  {
    std::string line;
    std::string prev_word;

    for (size_t i = 0; i < tokens.size(); ++i)
    {
      std::vector<std::string> parts = split_utf8(tokens[i], feature_marker);

      if (i > 0 && !has_right_join(prev_word) && !has_left_join(parts[0]))
        line += " ";

      std::string word = parts[0];

      if (has_right_join(word))
        word.erase(word.length() - _joiner.length(), _joiner.length());
      if (has_left_join(word))
        word.erase(0, _joiner.length());

      if (_case_feature)
      {
        if (parts.size() < 2)
          throw std::runtime_error("Missing case feature");
        word = CaseModifier::apply_case(word, parts[1][0]);
      }

      line += word;
      prev_word = parts[0];
    }

    return line;
  }

  std::vector<std::string> Tokenizer::tokenize_line(const std::string& text)
  {
    std::vector<std::string> chars;
    std::vector<unicode_code_point_t> code_points;

    explode_utf8(text, chars, code_points);

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

  bool Tokenizer::has_left_join(const std::string& word)
  {
    return (word.length() >= _joiner.length() && word.substr(0, _joiner.length()) == _joiner);
  }

  bool Tokenizer::has_right_join(const std::string& word)
  {
    return (word.length() >= _joiner.length()
            && word.substr(word.length() - _joiner.length(), _joiner.length()) == _joiner);
  }

}
