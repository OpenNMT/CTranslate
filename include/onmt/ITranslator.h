#pragma once

#include <string>
#include <vector>

#include "onmt/TranslationResult.h"

namespace onmt
{

  class ITranslator
  {
  public:
    virtual ~ITranslator() = default;
    virtual std::string translate(const std::string& text) = 0;
    virtual std::vector<std::string> translate_batch(const std::vector<std::string>& texts) = 0;

    virtual TranslationResult
    translate(const std::vector<std::string>& tokens,
              const std::vector<std::vector<std::string> >& features) = 0;

    virtual TranslationResult
    translate_batch(const std::vector<std::vector<std::string> >& batch_tokens,
                    const std::vector<std::vector<std::vector<std::string> > >& batch_features) = 0;
  };

}
