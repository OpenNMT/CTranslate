#pragma once

#include <memory>

#include "Translator.h"

namespace onmt
{

  class TranslatorFactory
  {
  public:
    static std::unique_ptr<ITranslator> build(const std::string& model,
                                              const std::string& phrase_table = "",
                                              const std::string& vocab_mapping = "",
                                              bool replace_unk = true,
                                              size_t max_sent_length = 250,
                                              size_t beam_size = 5,
                                              bool cuda = false,
                                              bool profiling = false);
  };

}
