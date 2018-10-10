#pragma once

#include <string>
#include <vector>

class BatchInput
{
public:
  BatchInput(const std::vector<std::string>& data, size_t id)
    : _data(data)
    , _id(id)
  {
  }

  size_t size() const
  {
    return _data.size();
  }

  bool empty() const
  {
    return _data.empty();
  }

  const std::vector<std::string>& get_data() const
  {
    return _data;
  }

  size_t get_id() const
  {
    return _id;
  }

private:
  std::vector<std::string> _data;
  size_t _id;
};

class BatchOutput
{
public:
  BatchOutput(const std::vector<std::vector<std::string> >& data, size_t id)
    : _data(data)
    , _id(id)
  {
  }

  size_t size() const
  {
    return _data.size();
  }

  bool empty() const
  {
    return _data.empty();
  }

  const std::vector<std::vector<std::string> >& get_data() const
  {
    return _data;
  }

  size_t get_id() const
  {
    return _id;
  }

private:
  std::vector<std::vector<std::string> > _data;
  size_t _id;
};
