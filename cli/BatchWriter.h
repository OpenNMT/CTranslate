#pragma once

#include <fstream>
#include <mutex>
#include <map>

#include "Batch.h"

class BatchWriter
{
public:
  BatchWriter(const std::string& file);
  BatchWriter(std::ostream& out);

  void write(const Batch& batch);
  size_t total_count_src_words() const;
  size_t total_count_src_unk_words() const;
  size_t total_count_tgt_words() const;
  size_t total_count_tgt_unk_words() const;
  float total_score() const;

private:
  std::ofstream _file;
  std::ostream& _out;
  size_t _sentence_no;
  std::map<size_t, Batch> _pending_batches;
  size_t _last_batch_id;
  size_t _total_count_src_words;
  size_t _total_count_src_unk_words;
  size_t _total_count_tgt_words;
  size_t _total_count_tgt_unk_words;
  float _total_score;
  std::mutex _writer_mutex;
};
