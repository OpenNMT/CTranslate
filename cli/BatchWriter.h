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

  void write(const BatchOutput& batch);

private:
  std::ofstream _file;
  std::ostream& _out;
  std::map<size_t, std::vector<std::vector<std::string>>> _pending_batches;
  size_t _last_batch_id;
  std::mutex _writer_mutex;
};
