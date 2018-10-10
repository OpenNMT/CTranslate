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
                                              size_t n_best = 1,
                                              bool cuda = false,
                                              bool qlinear = false,
                                              bool profiling = false);

    static std::unique_ptr<ITranslator> clone(const std::unique_ptr<ITranslator>& translator);
  };

}
