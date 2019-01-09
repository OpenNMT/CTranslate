#pragma once

#include <string>
#include <vector>

#include "onmt/onmt_export.h"

namespace onmt
{

  class ONMT_EXPORT TranslationResult
  {
  public:
    TranslationResult(const std::vector<std::vector<std::vector<std::string> > >& words,
                      const std::vector<std::vector<std::vector<std::vector<std::string> > > >& features,
                      const std::vector<std::vector<std::vector<std::vector<float> > > >& attention,
                      const std::vector<std::vector<float> >& score,
                      const std::vector<std::vector<size_t> >& count_unk_words);

    const std::vector<std::string>& get_words(size_t job_index = 0, size_t translation_index = 0) const;
    const std::vector<std::vector<std::string> >& get_features(size_t job_index = 0, size_t translation_index = 0) const;
    const std::vector<std::vector<float> >& get_attention(size_t job_index = 0, size_t translation_index = 0) const;
    float get_score(size_t job_index = 0, size_t translation_index = 0) const;
    size_t get_count_unk_words(size_t job_index = 0, size_t translation_index = 0) const;

    const std::vector<std::vector<std::string> >& get_words_job(size_t job_index = 0) const;
    const std::vector<std::vector<std::vector<std::string> > >& get_features_job(size_t job_index = 0) const;
    const std::vector<std::vector<std::vector<float> > >& get_attention_job(size_t job_index = 0) const;
    const std::vector<float>& get_score_job(size_t job_index = 0) const;
    const std::vector<size_t>& get_count_unk_words_job(size_t job_index = 0) const;
    size_t count_job(size_t job_index = 0) const;

    const std::vector<std::vector<std::vector<std::string> > >& get_words_batch() const;
    const std::vector<std::vector<std::vector<std::vector<std::string> > > >& get_features_batch() const;
    const std::vector<std::vector<std::vector<std::vector<float> > > >& get_attention_batch() const;
    const std::vector<std::vector<float> >& get_score_batch() const;
    const std::vector<std::vector<size_t> >& get_count_unk_words_batch() const;

    size_t count() const;
    bool has_features() const;

  private:
    std::vector<std::vector<std::vector<std::string> > > _words;
    std::vector<std::vector<std::vector<std::vector<std::string> > > > _features;
    std::vector<std::vector<std::vector<std::vector<float> > > > _attention;
    std::vector<std::vector<float> > _score;
    std::vector<std::vector<size_t> > _count_unk_words;
  };

}
