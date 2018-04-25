#include "BatchWriter.h"

BatchWriter::BatchWriter(const std::string& file)
  : _file(file.c_str())
  , _out(_file)
  , _last_batch_id(0)
{
}

BatchWriter::BatchWriter(std::ostream& out)
  : _out(out)
{
}

void BatchWriter::write(const Batch& batch)
{
  std::lock_guard<std::mutex> lock(_writerMutex);
  size_t batch_id = batch.get_id();
  if (!batch_id)
    return;
  if (batch_id == _last_batch_id+1)
  {
    std::vector<std::string> v = batch.get_data();
    while (batch_id == _last_batch_id + 1) {
      for (const auto& sent: v)
        _out << sent << std::endl;
      _last_batch_id = batch_id;
      auto it = _pending_batches.find(_last_batch_id + 1);
      if (it != _pending_batches.end())
      {
        v = it -> second;
        batch_id = _last_batch_id + 1;
        _pending_batches.erase(it);
      }
    }
  }
  else
    _pending_batches[batch_id] = batch.get_data();

}
