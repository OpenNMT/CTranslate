#pragma once

#include <fstream>
#include <mutex>

#include "Batch.h"

class BatchReader
{
public:
  BatchReader(const std::string& file, size_t batch_size);
  BatchReader(std::istream& in, size_t batch_size);

  Batch read_next();
  size_t size() const { return _read_size; }

private:
  std::ifstream _file;
  std::istream& _in;
  size_t _batch_size;
  int _batch_id;
  size_t _read_size;
  std::mutex _reader_mutex;
};
