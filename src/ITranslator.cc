#include "onmt/ITranslator.h"

#include "onmt/SpaceTokenizer.h"

namespace onmt
{

  std::string ITranslator::translate(const std::string& text)
  {
    return translate(text, SpaceTokenizer::get_instance());
  }

  std::string ITranslator::translate(const std::string& text, ITokenizer& tokenizer)
  {
    return get_translations(text, tokenizer).at(0);
  }

  std::vector<std::string> ITranslator::translate_batch(const std::vector<std::string>& texts)
  {
    return translate_batch(texts, SpaceTokenizer::get_instance());
  }

  std::vector<std::string> ITranslator::translate_batch(const std::vector<std::string>& texts, ITokenizer& tokenizer)
  {
    return get_translations_batch(texts, tokenizer).at(0);
  }

  std::vector<std::string> ITranslator::get_translations(const std::string& text)
  {
    return get_translations(text, SpaceTokenizer::get_instance());
  }

  std::vector<std::vector<std::string> > ITranslator::get_translations_batch(const std::vector<std::string>& texts)
  {
    return get_translations_batch(texts, SpaceTokenizer::get_instance());
  }

}
