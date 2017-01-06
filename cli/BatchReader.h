#pragma once

#include <fstream>
#include <string>
#include <vector>

class BatchReader
{
public:
  BatchReader(const std::string& file, size_t batch_size);
  BatchReader(std::istream& in, size_t batch_size);

  std::vector<std::string> read_next();

private:
  std::ifstream _file;
  std::istream& _in;
  size_t _batch_size;
};
