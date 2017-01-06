#include "BatchWriter.h"

BatchWriter::BatchWriter(const std::string& file)
  : _file(file.c_str())
  , _out(_file)
{
}

BatchWriter::BatchWriter(std::ostream& out)
  : _out(out)
{
}

void BatchWriter::write(const std::vector<std::string>& batch)
{
  for (const auto& sent: batch)
    _out << sent << std::endl;
}
