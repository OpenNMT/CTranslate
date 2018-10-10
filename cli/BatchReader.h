#pragma once

#include <fstream>
#include <mutex>

#include "Batch.h"

class BatchReader
{
public:
  BatchReader(const std::string& file, size_t batch_size);
  BatchReader(std::istream& in, size_t batch_size);

  BatchInput read_next();

  size_t read_lines() const
  {
    return _read_lines;
  }

private:
  std::ifstream _file;
  std::istream& _in;
  size_t _batch_size;
  size_t _batch_id;
  size_t _read_lines;
  std::mutex _reader_mutex;
};
