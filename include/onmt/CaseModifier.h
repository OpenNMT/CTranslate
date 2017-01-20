#pragma once

#include <string>
#include <unicode.h>

namespace onmt
{

  class CaseModifier
  {
  public:
    static char extract_case(const std::string& token,
                             std::string& normalized_token);

  private:
    enum class Type
    {
      Lowercase,
      Uppercase,
      Mixed,
      Capitalized,
      CapitalizedFirst,
      None
    };

    static Type update_type(Type current, _type_letter type);

    static char type_to_char(Type type);
    static Type char_to_type(char feature);
  };

}
