#pragma once

#include <string>
#include <vector>

class Batch
{
  public:
    Batch():_id(0) {}
    Batch(const std::vector<std::string> &data, int id):_data(data), _id(id) {}
    size_t size() const { return _data.size(); }
    bool empty() const { return _data.empty(); }
    const std::vector<std::string> &get_data() const { return _data; }
    size_t get_id() const { return _id; }
  private:
    std::vector<std::string> _data;
    size_t _id;
};
