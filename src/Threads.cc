#include "onmt/Threads.h"

#include <Eigen/Core>
#ifdef WITH_MKL
#  include <mkl.h>
#endif

namespace onmt
{

  void Threads::set(int number)
  {
    Eigen::setNbThreads(number);
#ifdef WITH_MKL
    mkl_set_num_threads(number);
#endif
  }

  int Threads::get()
  {
    return Eigen::nbThreads();
  }

}
