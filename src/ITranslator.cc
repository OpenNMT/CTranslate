#include "onmt/ITranslator.h"

#include "onmt/SpaceTokenizer.h"

namespace onmt
{

  std::string ITranslator::translate(const std::string& text)
  {
    return translate(text, SpaceTokenizer::get_instance());
  }

  std::vector<std::string> ITranslator::translate_batch(const std::vector<std::string>& texts)
  {
    return translate_batch(texts, SpaceTokenizer::get_instance());
  }

}
