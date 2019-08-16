#pragma once

#include <stddef.h>

#include "onmt/onmt_export.h"

namespace onmt
{

  class ONMT_EXPORT TranslationOptions
  {
  public:
    TranslationOptions(size_t max_sent_length = 250,
                       size_t beam_size = 5,
                       size_t n_best = 1,
                       bool replace_unk = true,
                       bool replace_unk_tagged = false);

    size_t max_sent_length() const;
    size_t& max_sent_length();
    size_t beam_size() const;
    size_t& beam_size();
    size_t n_best() const;
    size_t& n_best();
    bool replace_unk() const;
    bool& replace_unk();
    bool replace_unk_tagged() const;
    bool& replace_unk_tagged();

  private:
    size_t _max_sent_length;
    size_t _beam_size;
    size_t _n_best;
    bool _replace_unk;
    bool _replace_unk_tagged;
  };

}
