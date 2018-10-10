#pragma once

#include <string>
#include <vector>

namespace onmt
{

  class TranslationResult
  {
  public:
    TranslationResult(const std::vector<std::vector<std::vector<std::string> > >& words,
                      const std::vector<std::vector<std::vector<std::vector<std::string> > > >& features,
                      const std::vector<std::vector<std::vector<std::vector<float> > > >& attention);

    const std::vector<std::string>& get_words(size_t batch_index, size_t text_index) const;
    const std::vector<std::vector<std::string> >& get_features(size_t batch_index, size_t text_index) const;
    const std::vector<std::vector<float> >& get_attention(size_t batch_index, size_t text_index) const;

    const std::vector<std::vector<std::string> >& get_words(size_t batch_index) const;
    const std::vector<std::vector<std::vector<std::string> > >& get_features(size_t batch_index) const;
    const std::vector<std::vector<std::vector<float> > >& get_attention(size_t batch_index) const;
    size_t count(size_t batch_index) const;

    const std::vector<std::vector<std::vector<std::string> > >& get_words_batch() const;
    const std::vector<std::vector<std::vector<std::vector<std::string> > > >& get_features_batch() const;
    const std::vector<std::vector<std::vector<std::vector<float> > > >& get_attention_batch() const;

    size_t count_batch() const;
    bool has_features() const;

  private:
    std::vector<std::vector<std::vector<std::string> > > _words;
    std::vector<std::vector<std::vector<std::vector<std::string> > > > _features;
    std::vector<std::vector<std::vector<std::vector<float> > > > _attention;
  };

}
