#include "BatchWriter.h"
#include <onmt/Utils.h>
#include <iomanip>
#include <ios>

BatchWriter::BatchWriter(const std::string& file)
  : _file(file.c_str())
  , _out(_file)
  , _sentence_no(0)
  , _last_batch_id(0)
  , _total_count_src_words(0)
  , _total_count_src_unk_words(0)
  , _total_count_tgt_words(0)
  , _total_count_tgt_unk_words(0)
  , _total_score(0)
{
  if (!_file.is_open())
    ONMT_LOG_STREAM_SEV("cannot open '" << file << '\'', boost::log::trivial::error);
}

BatchWriter::BatchWriter(std::ostream& out)
  : _out(out)
  , _sentence_no(0)
  , _last_batch_id(0)
  , _total_count_src_words(0)
  , _total_count_src_unk_words(0)
  , _total_count_tgt_words(0)
  , _total_count_tgt_unk_words(0)
  , _total_score(0)
{
}

void BatchWriter::write(const Batch& batch)
{
  std::lock_guard<std::mutex> lock(_writer_mutex);
  size_t batch_id = batch.get_id();
  if (batch_id == _last_batch_id + 1)
  {
    auto input = batch.get_input();
    auto translations = batch.get_translations();
    auto score = batch.get_score();
    auto count_tgt_words = batch.get_count_tgt_words();
    auto count_tgt_unk_words = batch.get_count_tgt_unk_words();
    auto count_src_words = batch.get_count_src_words();
    auto count_src_unk_words = batch.get_count_src_unk_words();

    while (batch_id == _last_batch_id + 1)
    {
      for (size_t b = 0; b < input.size(); ++b)
      {
        _total_count_src_words += count_src_words[b];
        _total_count_src_unk_words += count_src_unk_words[b];
        ONMT_LOG_STREAM_SEV("SENT " << ++_sentence_no << ": " << input[b], boost::log::trivial::info);
        if (translations[b].size() > 1)
        {
          ONMT_LOG_STREAM_SEV("", boost::log::trivial::info);
          ONMT_LOG_STREAM_SEV("BEST HYP:", boost::log::trivial::info);
        }

        for (size_t n = 0; n < translations[b].size(); ++n)
        {
          _out << translations[b][n] << std::endl;
          if (translations[b].size() > 1)
          {
            ONMT_LOG_STREAM_SEV('[' << std::fixed << std::setprecision(2) << score[b][n] << "] " << translations[b][n], boost::log::trivial::info);
          }
          else
          {
            ONMT_LOG_STREAM_SEV("PRED " << _sentence_no << ": " << translations[b][n], boost::log::trivial::info);
            ONMT_LOG_STREAM_SEV("PRED SCORE: " << std::fixed << std::setprecision(2) << score[b][n], boost::log::trivial::info);
          }

          // count target unknown words and words generated on 1-best
          if (n == 0)
          {
            _total_count_tgt_words += count_tgt_words[b][n];
            _total_count_tgt_unk_words += count_tgt_unk_words[b][n];
            _total_score += score[b][n];
          }
        }

        ONMT_LOG_STREAM_SEV("", boost::log::trivial::info);
      }

      _last_batch_id = batch_id;
      auto it = _pending_batches.find(_last_batch_id + 1);
      if (it != _pending_batches.end())
      {
        input = it->second.get_input();
        translations = it->second.get_translations();
        score = it->second.get_score();
        count_tgt_words = it->second.get_count_tgt_words();
        count_tgt_unk_words = it->second.get_count_tgt_unk_words();
        count_src_words = it->second.get_count_src_words();
        count_src_unk_words = it->second.get_count_src_unk_words();
        batch_id = _last_batch_id + 1;
        _pending_batches.erase(it);
      }
    }
  }
  else
  {
    _pending_batches[batch_id] = batch;
  }
}

size_t BatchWriter::total_count_src_words() const
{
  return _total_count_src_words;
}

size_t BatchWriter::total_count_src_unk_words() const
{
  return _total_count_src_unk_words;
}

size_t BatchWriter::total_count_tgt_words() const
{
  return _total_count_tgt_words;
}

size_t BatchWriter::total_count_tgt_unk_words() const
{
  return _total_count_tgt_unk_words;
}

float BatchWriter::total_score() const
{
  return _total_score;
}
