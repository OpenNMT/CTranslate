#include "onmt/ITranslator.h"
#include "onmt/SpaceTokenizer.h"
#include <utility>

namespace onmt
{

  std::string
  ITranslator::translate(const std::string& text,
                         const TranslationOptions& options)
  {
    return translate(text, SpaceTokenizer::get_instance(), options);
  }

  std::string
  ITranslator::translate(const std::string& text,
                         float& score,
                         size_t& count_tgt_words,
                         size_t& count_tgt_unk_words,
                         size_t& count_src_words,
                         size_t& count_src_unk_words,
                         const TranslationOptions& options)
  {
    return translate(text, SpaceTokenizer::get_instance(), score, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  std::string
  ITranslator::translate(const std::string& text,
                         ITokenizer& tokenizer,
                         const TranslationOptions& options)
  {
    float score;
    size_t count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words;
    return translate(text, tokenizer, score, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  std::string
  ITranslator::translate(const std::string& text,
                         ITokenizer& tokenizer,
                         float& score,
                         size_t& count_tgt_words,
                         size_t& count_tgt_unk_words,
                         size_t& count_src_words,
                         size_t& count_src_unk_words,
                         const TranslationOptions& options)
  {
    std::vector<float> best_scores;
    std::vector<size_t> best_count_tgt_words, best_count_tgt_unk_words;
    auto res = get_translations(text, tokenizer, best_scores, best_count_tgt_words, best_count_tgt_unk_words, count_src_words, count_src_unk_words, options);
    score = best_scores.at(0);
    count_tgt_words = best_count_tgt_words.at(0);
    count_tgt_unk_words = best_count_tgt_unk_words.at(0);
    return res.at(0);
  }

  std::vector<std::string>
  ITranslator::get_translations(const std::string& text,
                                const TranslationOptions& options)
  {
    return get_translations(text, SpaceTokenizer::get_instance(), options);
  }

  std::vector<std::string>
  ITranslator::get_translations(const std::string& text,
                                std::vector<float>& scores,
                                std::vector<size_t>& count_tgt_words,
                                std::vector<size_t>& count_tgt_unk_words,
                                size_t& count_src_words,
                                size_t& count_src_unk_words,
                                const TranslationOptions& options)
  {
    return get_translations(text, SpaceTokenizer::get_instance(), scores, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  std::vector<std::string>
  ITranslator::get_translations(const std::string& text,
                                ITokenizer& tokenizer,
                                const TranslationOptions& options)
  {
    std::vector<float> scores;
    std::vector<size_t> count_tgt_words, count_tgt_unk_words;
    size_t count_src_words, count_src_unk_words;
    return get_translations(text, tokenizer, scores, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  TranslationResult
  ITranslator::translate(const std::vector<std::string>& tokens,
                         const std::vector<std::vector<std::string> >& features,
                         const TranslationOptions& options)
  {
    size_t count_src_unk_words;
    return translate(tokens, features, count_src_unk_words, options);
  }

  std::vector<std::string>
  ITranslator::translate_batch(const std::vector<std::string>& texts,
                               const TranslationOptions& options)
  {
    return translate_batch(texts, SpaceTokenizer::get_instance(), options);
  }

  std::vector<std::string>
  ITranslator::translate_batch(const std::vector<std::string>& texts,
                               std::vector<float>& scores,
                               std::vector<size_t>& count_tgt_words,
                               std::vector<size_t>& count_tgt_unk_words,
                               std::vector<size_t>& count_src_words,
                               std::vector<size_t>& count_src_unk_words,
                               const TranslationOptions& options)
  {
    return translate_batch(texts, SpaceTokenizer::get_instance(), scores, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  std::vector<std::string>
  ITranslator::translate_batch(const std::vector<std::string>& texts,
                               ITokenizer& tokenizer,
                               const TranslationOptions& options)
  {
    std::vector<float> scores;
    std::vector<size_t> count_tgt_words, count_tgt_unk_words;
    std::vector<size_t> count_src_words, count_src_unk_words;
    return translate_batch(texts, tokenizer, scores, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  std::vector<std::string>
  ITranslator::translate_batch(const std::vector<std::string>& texts,
                               ITokenizer& tokenizer,
                               std::vector<float>& scores,
                               std::vector<size_t>& count_tgt_words,
                               std::vector<size_t>& count_tgt_unk_words,
                               std::vector<size_t>& count_src_words,
                               std::vector<size_t>& count_src_unk_words,
                               const TranslationOptions& options)
  {
    std::vector<std::string> translations;
    scores.clear();
    count_tgt_words.clear();
    count_tgt_unk_words.clear();
    std::vector<std::vector<float> > batch_scores;
    std::vector<std::vector<size_t> > batch_count_tgt_words, batch_count_tgt_unk_words;
    auto res = get_translations_batch(texts, tokenizer, batch_scores, batch_count_tgt_words, batch_count_tgt_unk_words, count_src_words, count_src_unk_words, options);
    for (size_t i = 0; i < res.size(); ++i)
    {
      translations.push_back(std::move(res[i].at(0)));
      scores.push_back(batch_scores[i].at(0));
      count_tgt_words.push_back(batch_count_tgt_words[i].at(0));
      count_tgt_unk_words.push_back(batch_count_tgt_unk_words[i].at(0));
    }

    return translations;
  }

  std::vector<std::vector<std::string> >
  ITranslator::get_translations_batch(const std::vector<std::string>& texts,
                                      const TranslationOptions& options)
  {
    return get_translations_batch(texts, SpaceTokenizer::get_instance(), options);
  }

  std::vector<std::vector<std::string> >
  ITranslator::get_translations_batch(const std::vector<std::string>& texts,
                                      std::vector<std::vector<float> >& scores,
                                      std::vector<std::vector<size_t> >& count_tgt_words,
                                      std::vector<std::vector<size_t> >& count_tgt_unk_words,
                                      std::vector<size_t>& count_src_words,
                                      std::vector<size_t>& count_src_unk_words,
                                      const TranslationOptions& options)
  {
    return get_translations_batch(texts, SpaceTokenizer::get_instance(), scores, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  std::vector<std::vector<std::string> >
  ITranslator::get_translations_batch(const std::vector<std::string>& texts,
                                      ITokenizer& tokenizer,
                                      const TranslationOptions& options)
  {
    std::vector<std::vector<float> > scores;
    std::vector<std::vector<size_t> > count_tgt_words, count_tgt_unk_words;
    std::vector<size_t> count_src_words, count_src_unk_words;
    return get_translations_batch(texts, tokenizer, scores, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words, options);
  }

  TranslationResult
  ITranslator::translate_batch(const std::vector<std::vector<std::string> >& batch_tokens,
                               const std::vector<std::vector<std::vector<std::string> > >& batch_features,
                               const TranslationOptions& options)
  {
    std::vector<size_t> batch_count_src_unk_words;
    return translate_batch(batch_tokens, batch_features, batch_count_src_unk_words, options);
  }

}
