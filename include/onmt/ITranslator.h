#pragma once

#include <string>
#include <vector>

#include "onmt/ITokenizer.h"
#include "onmt/TranslationResult.h"

namespace onmt
{

  class ITranslator
  {
  public:
    virtual ~ITranslator() = default;

    // Translate a raw text. If the tokenizer is not given, the input text is split on spaces.
    virtual std::string
    translate(const std::string& text);
    virtual std::string
    translate(const std::string& text, ITokenizer& tokenizer) = 0;

    // Translate pre-tokenized text.
    virtual TranslationResult
    translate(const std::vector<std::string>& tokens,
              const std::vector<std::vector<std::string> >& features) = 0;


    // Batch version of the previous methods: translate several sequences at once.

    virtual std::vector<std::string>
    translate_batch(const std::vector<std::string>& texts);
    virtual std::vector<std::string>
    translate_batch(const std::vector<std::string>& texts, ITokenizer& tokenizer) = 0;

    virtual TranslationResult
    translate_batch(const std::vector<std::vector<std::string> >& batch_tokens,
                    const std::vector<std::vector<std::vector<std::string> > >& batch_features) = 0;
  };

}
