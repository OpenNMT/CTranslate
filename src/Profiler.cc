#include "onmt/Profiler.h"

#include <algorithm>
#include <vector>

namespace onmt
{
  int Profiler::_counter = 0;
  std::mutex Profiler::_profiler_mutex;

  Profiler::Profiler(bool enabled, bool start_chrono)
    : _enabled(enabled)
    , _total_time(std::chrono::microseconds::zero())
  {
    if (start_chrono)
      start();
    reset();
    /* assign unique id */
    std::lock_guard<std::mutex> lock(_profiler_mutex);
    _id = ++_counter;
  }

  Profiler::~Profiler()
  {
    if (_enabled)
    {
      std::cerr << *this;
    }
  }

  void Profiler::enable()
  {
    _enabled = true;
  }

  void Profiler::disable()
  {
    _enabled = false;
  }

  void Profiler::reset()
  {
    _cumulated.clear();
  }

  int Profiler::get_id() const
  {
    return _id;
  }

  void Profiler::start()
  {
    if (_enabled)
    {
      _start.emplace(std::chrono::high_resolution_clock::now());
    }
  }

  void Profiler::stop(const std::string& module_name)
  {
    if (_enabled)
    {
      auto diff = std::chrono::high_resolution_clock::now() - _start.top();
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(diff);
      _total_time += elapsed;
      _start.pop();
      _cumulated[module_name] += elapsed;
    }
  }

  std::ostream& operator<<(std::ostream& os, const Profiler& profiler)
  {
      std::lock_guard<std::mutex> lock(Profiler::_profiler_mutex);
    // Sort accumulated time.
    std::vector<std::pair<std::string, std::chrono::microseconds> > samples;
    for (const auto& sample: profiler._cumulated)
      samples.emplace_back(sample);

    std::sort(samples.begin(), samples.end(),
              [] (const std::pair<std::string, std::chrono::microseconds>& a,
                  const std::pair<std::string, std::chrono::microseconds>& b)
              {
                return a.second > b.second;
              });

    for (auto it: samples)
    {
      os << "[" << profiler.get_id() << "]" 
         << '\t'
         << it.first
         << '\t'
         << static_cast<double>(it.second.count()) / 1000 << "ms"
         << '\t'
         << "(" << (static_cast<double>(it.second.count()) / static_cast<double>(profiler._total_time.count())) * 100 << "%)"
         << std::endl;
    }

    return os;
  }

}
