#include "onmt/Tokenizer.h"

#include <boost/algorithm/string.hpp>

#include "onmt/CaseModifier.h"

namespace onmt
{

  const std::string Tokenizer::joiner_marker("￭");

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

  std::string Tokenizer::detokenize(const std::vector<std::string>& words,
                                    const std::vector<std::vector<std::string> >& features)
  {
    std::string line;

    for (size_t i = 0; i < words.size(); ++i)
    {
      if (i > 0 && !has_right_join(words[i - 1]) && !has_left_join(words[i]))
        line += " ";

      std::string word = words[i];

      if (has_right_join(word))
        word.erase(word.length() - _joiner.length(), _joiner.length());
      if (has_left_join(word))
        word.erase(0, _joiner.length());

      if (_case_feature)
      {
        if (features.empty())
          throw std::runtime_error("Missing case feature");
        word = CaseModifier::apply_case(word, features[0][i][0]);
      }

      line += word;
    }

    return line;
  }

  void Tokenizer::tokenize(const std::string& text,
                           std::vector<std::string>& words,
                           std::vector<std::vector<std::string> >& features)
  {
    std::vector<std::string> chars;
    std::vector<unicode::code_point_t> code_points;

    unicode::explode_utf8(text, chars, code_points);

    std::string token;

    bool letter = false;
    bool number = false;
    bool other = false;
    bool space = true;

    unicode::_type_letter type_letter;

    for (size_t i = 0; i < chars.size(); ++i)
    {
      const std::string& c = chars[i];
      unicode::code_point_t v = code_points[i];
      unicode::code_point_t next_v = i + 1 < code_points.size() ? code_points[i + 1] : 0;

      if (unicode::is_separator(v))
      {
        if (!space)
        {
          words.push_back(token);
          token.clear();
        }

        if (v == 0x200D && _joiner_annotate)
        {
          if (_joiner_new && !words.empty())
            words.push_back(_joiner);
          else
          {
            if (other || (number && unicode::is_letter(next_v, type_letter)))
              words.back() += _joiner;
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
          cur_letter = unicode::is_letter(v, type_letter);
          cur_number = unicode::is_number(v);

          if (_mode == Mode::Conservative)
          {
            if (unicode::is_number(v)
                || (c == "-" && letter)
                || (c == "_")
                || (letter && (c == "." || c == ",") && (unicode::is_number(next_v) || unicode::is_letter(next_v, type_letter))))
              cur_letter = true;
          }

          if (cur_letter)
          {
            if (!letter && !space)
            {
              if (_joiner_annotate && !_joiner_new)
                token += _joiner;
              words.push_back(token);
              if (_joiner_annotate && _joiner_new)
                words.push_back(_joiner);
              token.clear();
            }
            else if (other && _joiner_annotate && token.empty())
            {
              if (_joiner_new)
                words.push_back(_joiner);
              else
                words.back() += _joiner;
            }

            token += c;
            letter = true;
            number = false;
            other = false;
            space = false;
          }
          else if (cur_number)
          {
            if (!number && !space)
            {
              if (_joiner_annotate && !_joiner_new && !letter)
                token += _joiner;
              words.push_back(token);
              if (_joiner_annotate && _joiner_new)
                words.push_back(_joiner);
              token.clear();
              if (_joiner_annotate && !_joiner_new && letter)
                token += _joiner;
            }
            else if (other && _joiner_annotate)
            {
              if (_joiner_new)
                words.push_back(_joiner);
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
              words.push_back(token);
              if (_joiner_annotate && _joiner_new)
                words.push_back(_joiner);
              token.clear();
              if (_joiner_annotate && !_joiner_new)
                token += _joiner;
            }
            else if (other && _joiner_annotate)
            {
              if (_joiner_new)
                words.push_back(_joiner);
              else
                token = _joiner;
            }

            token += c;
            words.push_back(token);
            token.clear();
            letter = false;
            number = false;
            other = true;
            space = true;
          }
        }
      }
    }

    if (!token.empty())
      words.push_back(token);

    if (_case_feature)
    {
      std::vector<std::string> case_feat;

      for (size_t i = 0; i < words.size(); ++i)
      {
        auto data = CaseModifier::extract_case(words[i]);
        words[i] = data.first;
        case_feat.emplace_back(1, data.second);
      }

      features.push_back(case_feat);
    }
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
