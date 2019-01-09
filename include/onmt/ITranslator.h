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
    translate(const std::string& text,
              float& score,
              size_t& count_tgt_words,
              size_t& count_tgt_unk_words,
              size_t& count_src_words,
              size_t& count_src_unk_words);
    virtual std::string
    translate(const std::string& text,
              ITokenizer& tokenizer);
    virtual std::string
    translate(const std::string& text,
              ITokenizer& tokenizer,
              float& score,
              size_t& count_tgt_words,
              size_t& count_tgt_unk_words,
              size_t& count_src_words,
              size_t& count_src_unk_words);

    // Multiple translations version of the previous methods.

    virtual std::vector<std::string>
    get_translations(const std::string& text);
    virtual std::vector<std::string>
    get_translations(const std::string& text,
                     std::vector<float>& scores,
                     std::vector<size_t>& count_tgt_words,
                     std::vector<size_t>& count_tgt_unk_words,
                     size_t& count_src_words,
                     size_t& count_src_unk_words);
    virtual std::vector<std::string>
    get_translations(const std::string& text,
                     ITokenizer& tokenizer);
    virtual std::vector<std::string>
    get_translations(const std::string& text,
                     ITokenizer& tokenizer,
                     std::vector<float>& scores,
                     std::vector<size_t>& count_tgt_words,
                     std::vector<size_t>& count_tgt_unk_words,
                     size_t& count_src_words,
                     size_t& count_src_unk_words) = 0;

    // Translate pre-tokenized text. (As with previous methods, this is also for multiple translations.)
    virtual TranslationResult
    translate(const std::vector<std::string>& tokens,
              const std::vector<std::vector<std::string> >& features);
    virtual TranslationResult
    translate(const std::vector<std::string>& tokens,
              const std::vector<std::vector<std::string> >& features,
              size_t& count_src_unk_words) = 0;


    // Batch version of the previous methods: translate several sequences at once.

    virtual std::vector<std::string>
    translate_batch(const std::vector<std::string>& texts);
    virtual std::vector<std::string>
    translate_batch(const std::vector<std::string>& texts,
                    std::vector<float>& scores,
                    std::vector<size_t>& count_tgt_words,
                    std::vector<size_t>& count_tgt_unk_words,
                    std::vector<size_t>& count_src_words,
                    std::vector<size_t>& count_src_unk_words);
    virtual std::vector<std::string>
    translate_batch(const std::vector<std::string>& texts,
                    ITokenizer& tokenizer);
    virtual std::vector<std::string>
    translate_batch(const std::vector<std::string>& texts,
                    ITokenizer& tokenizer,
                    std::vector<float>& scores,
                    std::vector<size_t>& count_tgt_words,
                    std::vector<size_t>& count_tgt_unk_words,
                    std::vector<size_t>& count_src_words,
                    std::vector<size_t>& count_src_unk_words);

    // Multiple translations version of the previous methods.

    virtual std::vector<std::vector<std::string> >
    get_translations_batch(const std::vector<std::string>& texts);
    virtual std::vector<std::vector<std::string> >
    get_translations_batch(const std::vector<std::string>& texts,
                           std::vector<std::vector<float> >& scores,
                           std::vector<std::vector<size_t> >& count_tgt_words,
                           std::vector<std::vector<size_t> >& count_tgt_unk_words,
                           std::vector<size_t>& count_src_words,
                           std::vector<size_t>& count_src_unk_words);
    virtual std::vector<std::vector<std::string> >
    get_translations_batch(const std::vector<std::string>& texts,
                           ITokenizer& tokenizer);
    virtual std::vector<std::vector<std::string> >
    get_translations_batch(const std::vector<std::string>& texts,
                           ITokenizer& tokenizer,
                           std::vector<std::vector<float> >& scores,
                           std::vector<std::vector<size_t> >& count_tgt_words,
                           std::vector<std::vector<size_t> >& count_tgt_unk_words,
                           std::vector<size_t>& count_src_words,
                           std::vector<size_t>& count_src_unk_words) = 0;

    // Translate pre-tokenized text. (As with previous methods, this is also for multiple translations.)
    virtual TranslationResult
    translate_batch(const std::vector<std::vector<std::string> >& batch_tokens,
                    const std::vector<std::vector<std::vector<std::string> > >& batch_features);
    virtual TranslationResult
    translate_batch(const std::vector<std::vector<std::string> >& batch_tokens,
                    const std::vector<std::vector<std::vector<std::string> > >& batch_features,
                    std::vector<size_t>& batch_count_src_unk_words) = 0;
  };

}
