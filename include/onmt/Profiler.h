#pragma once

#include <chrono>
#include <iostream>
#include <stack>
#include <string>
#include <unordered_map>
#include <mutex>

namespace onmt
{

  class Profiler
  {
  public:
    Profiler(bool enabled = false, bool start_chrono = false);
    ~Profiler();

    void enable();
    void disable();
    void reset();

    int get_id() const;

    void start();
    void stop(const std::string& module_name);

    friend std::ostream& operator<<(std::ostream& os, const Profiler& profiler);

  private:
    bool _enabled;
    std::chrono::microseconds _total_time;
    std::stack<std::chrono::high_resolution_clock::time_point> _start;
    std::unordered_map<std::string, std::chrono::microseconds> _cumulated;
    int _id;

    static std::mutex _profiler_mutex;
    static int _counter;
  };

}
