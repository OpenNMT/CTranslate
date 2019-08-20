#include "onmt/TranslationOptions.h"

namespace onmt
{

  TranslationOptions::TranslationOptions(size_t max_sent_length,
                                         size_t beam_size,
                                         size_t n_best,
                                         bool replace_unk,
                                         bool replace_unk_tagged)
    : _max_sent_length(max_sent_length)
    , _beam_size(beam_size)
    , _n_best(n_best)
    , _replace_unk(replace_unk)
    , _replace_unk_tagged(replace_unk_tagged)
  {
  }

  size_t TranslationOptions::max_sent_length() const
  {
    return _max_sent_length;
  }

  size_t& TranslationOptions::max_sent_length()
  {
    return _max_sent_length;
  }

  size_t TranslationOptions::beam_size() const
  {
    return _beam_size;
  }

  size_t& TranslationOptions::beam_size()
  {
    return _beam_size;
  }

  size_t TranslationOptions::n_best() const
  {
    return _n_best;
  }

  size_t& TranslationOptions::n_best()
  {
    return _n_best;
  }

  bool TranslationOptions::replace_unk() const
  {
    return _replace_unk;
  }

  bool& TranslationOptions::replace_unk()
  {
    return _replace_unk;
  }

  bool TranslationOptions::replace_unk_tagged() const
  {
    return _replace_unk_tagged;
  }

  bool& TranslationOptions::replace_unk_tagged()
  {
    return _replace_unk_tagged;
  }

}
