#include "BatchReader.h"

BatchReader::BatchReader(const std::string& file, size_t batch_size)
  : _file(file.c_str())
  , _in(_file)
  , _batch_size(batch_size)
  , _batch_id(0)
  , _read_size(0)
{
}

BatchReader::BatchReader(std::istream& in, size_t batch_size)
  : _in(in)
  , _batch_size(batch_size)
  , _batch_id(0)
  , _read_size(0)
{
}

Batch BatchReader::read_next()
{
  std::lock_guard<std::mutex> lock(_reader_mutex);
  std::vector<std::string> batch;

  if (_in.eof())
    return Batch(batch, ++_batch_id);

  batch.reserve(_batch_size);

  std::string line;
  while (batch.size() < _batch_size && std::getline(_in, line))
    batch.push_back(line);

  _read_size += batch.size();

  return Batch(batch, ++_batch_id);
}
