#pragma once

#include <memory>

#include "onmt/onmt_export.h"
#include "Translator.h"

namespace onmt
{

  class ONMT_EXPORT TranslatorFactory
  {
  public:
    static std::unique_ptr<ITranslator> build(const std::string& model,
                                              const std::string& phrase_table = "",
                                              const std::string& vocab_mapping = "",
                                              bool cuda = false,
                                              bool qlinear = false,
                                              bool profiling = false);

    static std::unique_ptr<ITranslator> clone(const std::unique_ptr<ITranslator>& translator);
  };

}
