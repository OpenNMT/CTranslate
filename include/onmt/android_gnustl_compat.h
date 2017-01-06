#pragma once

#ifdef ANDROID_GNUSTL_COMPAT

#include <string>
#include <sstream>
#include <cstdlib>

namespace std
{
  template<typename T>
  std::string to_string(T value)
  {
    std::ostringstream os;
    os << value;
    return os.str();
  }

  inline unsigned long stoul(const std::string& s)
  {
    return atol(s.c_str());
  }
}

#endif
