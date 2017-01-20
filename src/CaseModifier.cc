#include "onmt/CaseModifier.h"

namespace onmt
{

  char CaseModifier::extract_case(const std::string& token,
                                  std::string& normalized_token)
  {
    std::vector<std::string> chars;
    std::vector<unicode_code_point_t> code_points;

    split_utf8(token, chars, code_points);

    Type current_case = Type::None;
    std::string new_token;

    for (size_t i = 0; i < chars.size(); ++i)
    {
      unicode_code_point_t v = code_points[i];
      _type_letter type_letter;

      if (is_letter(v, type_letter))
      {
        current_case = update_type(current_case, type_letter);
        unicode_code_point_t lower = get_lower(v);
        if (lower)
          v = lower;
      }

      normalized_token += cp_to_utf8(v);
    }

    return type_to_char(current_case);
  }

  CaseModifier::Type CaseModifier::update_type(Type current, _type_letter type)
  {
    switch (current)
    {
    case Type::None:
      if (type == _letter_lower)
        return Type::Lowercase;
      if (type == _letter_upper)
        return Type::CapitalizedFirst;
      break;
    case Type::Lowercase:
      if (type == _letter_upper)
        return Type::Mixed;
      break;
    case Type::CapitalizedFirst:
      if (type == _letter_lower)
        return Type::Capitalized;
      if (type == _letter_upper)
        return Type::Uppercase;
      break;
    case Type::Capitalized:
      if (type == _letter_upper)
        return Type::Mixed;
      break;
    case Type::Uppercase:
      if (type == _letter_lower)
        return Type::Mixed;
      break;
    default:
      break;
    }

    return current;
  }

  char CaseModifier::type_to_char(Type type)
  {
    switch (type)
    {
    case Type::Lowercase:
      return 'L';
    case Type::Uppercase:
      return 'U';
    case Type::Mixed:
      return 'M';
    case Type::Capitalized:
    case Type::CapitalizedFirst:
      return 'C';
    default:
      return 'N';
    }
  }

  CaseModifier::Type CaseModifier::char_to_type(char feature)
  {
    switch (feature)
    {
    case 'L':
      return Type::Lowercase;
    case 'U':
      return Type::Uppercase;
    case 'M':
      return Type::Mixed;
    case 'C':
      return Type::Capitalized;
    default:
      return Type::None;
    }
  }

}
