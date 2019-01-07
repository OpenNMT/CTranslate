#pragma once

#include <string>
#include <vector>

class Batch
{
public:
  Batch()
    : _id(0)
  {
  }

  Batch(const std::vector<std::string>& input, size_t id)
    : _input(input)
    , _id(id)
  {
  }

  void set_result(const std::vector<std::vector<std::string> >& translations,
                  const std::vector<std::vector<float> >& score,
                  const std::vector<std::vector<size_t> >& count_tgt_words,
                  const std::vector<std::vector<size_t> >& count_tgt_unk_words,
                  const std::vector<size_t>& count_src_words,
                  const std::vector<size_t>& count_src_unk_words)
  {
    _translations = translations;
    _score = score;
    _count_tgt_words = count_tgt_words;
    _count_tgt_unk_words = count_tgt_unk_words;
    _count_src_words = count_src_words;
    _count_src_unk_words = count_src_unk_words;
  }

  size_t size() const
  {
    return _input.size();
  }

  bool empty() const
  {
    return _input.empty();
  }

  const std::vector<std::string>& get_input() const
  {
    return _input;
  }

  const std::vector<std::vector<std::string> >& get_translations() const
  {
    return _translations;
  }

  const std::vector<std::vector<float> >& get_score() const
  {
    return _score;
  }

  const std::vector<std::vector<size_t> >& get_count_tgt_words() const
  {
    return _count_tgt_words;
  }

  const std::vector<std::vector<size_t> >& get_count_tgt_unk_words() const
  {
    return _count_tgt_unk_words;
  }

  const std::vector<size_t>& get_count_src_words() const
  {
    return _count_src_words;
  }

  const std::vector<size_t>& get_count_src_unk_words() const
  {
    return _count_src_unk_words;
  }

  size_t get_id() const
  {
    return _id;
  }

private:
  std::vector<std::string> _input;
  std::vector<std::vector<std::string> > _translations;
  std::vector<std::vector<float> > _score;
  std::vector<size_t> _count_src_words;
  std::vector<size_t> _count_src_unk_words;
  std::vector<std::vector<size_t> > _count_tgt_words;
  std::vector<std::vector<size_t> > _count_tgt_unk_words;
  size_t _id;
};
