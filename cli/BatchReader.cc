#include "BatchReader.h"

BatchReader::BatchReader(const std::string& file, size_t batch_size)
  : _file(file.c_str())
  , _in(_file)
  , _batch_size(batch_size)
{
}

BatchReader::BatchReader(std::istream& in, size_t batch_size)
  : _in(in)
  , _batch_size(batch_size)
{
}

std::vector<std::string> BatchReader::read_next()
{
  std::vector<std::string> batch;
  batch.reserve(_batch_size);

  std::string line;

  while (batch.size() < _batch_size && std::getline(_in, line))
    batch.push_back(line);

  return batch;
}
