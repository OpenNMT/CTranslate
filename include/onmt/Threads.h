#pragma once

#include "onmt_export.h"

namespace onmt
{

  class ONMT_EXPORT Threads
  {
  public:
    static void set(int number);
    static int get();
  };

}
