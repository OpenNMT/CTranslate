#include "onmt/TranslationResult.h"

namespace onmt
{

  TranslationResult::TranslationResult(const std::vector<std::vector<std::string> >& words,
                                       const std::vector<std::vector<std::vector<std::string> > >& features,
                                       const std::vector<std::vector<std::vector<float> > >& attention)
    : _words(words)
    , _features(features)
    , _attention(attention)
  {
  }

  const std::vector<std::string>& TranslationResult::get_words(size_t index) const
  {
    return _words[index];
  }

  const std::vector<std::vector<std::string> >& TranslationResult::get_features(size_t index) const
  {
    return _features[index];
  }

  const std::vector<std::vector<float> >& TranslationResult::get_attention(size_t index) const
  {
    return _attention[index];
  }

  const std::vector<std::vector<std::string> >& TranslationResult::get_words_batch() const
  {
    return _words;
  }

  const std::vector<std::vector<std::vector<std::string> > >& TranslationResult::get_features_batch() const
  {
    return _features;
  }

  const std::vector<std::vector<std::vector<float> > >& TranslationResult::get_attention_batch() const
  {
    return _attention;
  }

  size_t TranslationResult::count() const
  {
    return _words.size();
  }

  bool TranslationResult::has_features() const
  {
    return !_features.empty();
  }

}
