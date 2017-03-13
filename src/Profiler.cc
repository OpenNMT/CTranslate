#include "onmt/Profiler.h"

#include <algorithm>
#include <vector>

namespace onmt
{

  Profiler::Profiler(bool enabled)
    : _enabled(enabled)
  {
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
    _total_time = std::chrono::microseconds::zero();
    _cumulated.clear();
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
      os << it.first
         << '\t'
         << static_cast<double>(it.second.count()) / 1000 << "ms"
         << '\t'
         << "(" << (static_cast<double>(it.second.count()) / static_cast<double>(profiler._total_time.count())) * 100 << "%)"
         << std::endl;
    }

    return os;
  }

}
