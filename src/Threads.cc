#include "onmt/Threads.h"

#include <Eigen/Core>

namespace onmt
{

  void Threads::set(int number)
  {
    return Eigen::setNbThreads(number);
  }

  int Threads::get()
  {
    return Eigen::nbThreads();
  }

}
